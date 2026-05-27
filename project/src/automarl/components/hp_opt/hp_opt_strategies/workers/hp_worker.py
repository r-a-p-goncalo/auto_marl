import os
import threading
from automarl.components.basic_components.exec_component import State, StopExperiment
from automarl.components.basic_components.stateful_component_loder import StatefulComponentLoader
from automarl.component import Component
from automarl.components.hp_opt.hp_opt_strategies.hp_optimization_loader import HyperparameterOptimizationLoader
from automarl.components.hp_opt.hp_optimization_pipeline import HyperparameterOptimizationPipeline, Component_to_opt_type
from automarl.components.loggers.logger_component import override_first_logger
from automarl.components.loggers.result_logger import ResultLogger
import optuna
import pandas

import time


class HyperparameterOptimizationWorkerIndexed():
    

    def __init__(self, parent_hp_pipeline : HyperparameterOptimizationLoader, thread_index, 
    only_report_with_enough_runs=False, report_max=True):
        
        self.parent_hp_pipeline : HyperparameterOptimizationLoader = parent_hp_pipeline

        self.parent_hp_pipeline.lg.process_input_if_not_processed()
        self.thread_logger = self.parent_hp_pipeline.lg.clone(input_for_clone={"name" : f"{thread_index}", 
                                                                               "base_directory" : self.parent_hp_pipeline,
                                                                               "artifact_relative_directory" : "thread_loggers",
                                                                               "create_new_directory" : False,
                                                                               "log_text_file" : f"thread_logger_{thread_index}.txt"
                                                                               })
        
        self.thread_logger.writeLine(f"Worker was setup\n")

        self.thread_index = thread_index

        self._is_busy = False

        self._is_busy_sem = threading.RLock()

        self.only_report_with_enough_runs = only_report_with_enough_runs

        self.report_max = report_max

    def is_busy(self):

        with self._is_busy_sem:
            return self._is_busy

    def aquire_if_not_busy(self):

        with self._is_busy_sem:

            if not self._is_busy:
                self._is_busy = True
                return self
        
        return None
    
    def free_worker(self):

        with self._is_busy_sem:
            self._is_busy = False

    def check_if_should_stop_execution_earlier(self, trial):
        try:
            self.parent_hp_pipeline.check_if_should_stop_execution_earlier()

        except StopExperiment as e:
            self.thread_logger.writeLine(f"Worker noticed a signal to prematurely stop the process on trial {trial.number}, ending it...")
            raise e



    # LOADER RELATED ------------------------------------------------------------



    # OPTIMIZATION -------------------------------------------------------------------------
        
    
    
    def try_run_component_in_group_in_not_busy_all_steps(self,  trial : optuna.Trial, component_index, n_steps, component_loader : StatefulComponentLoader, should_end):

        with self._is_busy_sem:

            if self._is_busy:
                return None
    
            self._is_busy = True

        try:
            to_return = self.try_run_component_in_group_all_steps(trial, component_index, n_steps, component_loader)

        finally:
            with self._is_busy_sem:
                self._is_busy = False

        return to_return

    def try_run_component_in_group_in_not_busy(self,  trial : optuna.Trial, component_index, step, component_loader : StatefulComponentLoader):

        with self._is_busy_sem:

            if self._is_busy:
                return False
    
            self._is_busy = True


        with override_first_logger(self.thread_logger):
            to_return = self.parent_hp_pipeline.try_run_component_in_group(trial, component_index, step, component_loader)

        with self._is_busy_sem:
            self._is_busy = False

        return True

        

    def try_run_component_in_group_all_steps(self, trial : optuna.Trial, component_index, n_steps, component_loader : StatefulComponentLoader, should_end):

        self.check_if_should_stop_execution_earlier(trial) # check if experiment should be stopped earlier

        with override_first_logger(self.thread_logger):
            last_reported_step = self.parent_hp_pipeline.get_last_reported_step(trial, component_index)

        # IF we are to do an initial evaluation
        if last_reported_step < 0 and self.parent_hp_pipeline.do_initial_evaluation:

            with override_first_logger(self.thread_logger):
                self.parent_hp_pipeline.exec_do_initial_evaluation(trial, component_index, component_loader)
            

        if last_reported_step < 0:
            next_step = 1
        else:
            next_step = last_reported_step + 1

        self.thread_logger.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEPS TO DO {n_steps} -----------------------\n")
            
        to_return = None

        for step in range(next_step, next_step + n_steps):

            if should_end[0]:
                self.thread_logger.writeLine(f"Interruped in trial {trial.number}, in component {component_index}, before step {step}, due to concurrent logic (probably some evaluation made it so the whole trial was pruned)")
                break

            self.check_if_should_stop_execution_earlier(trial)

            with override_first_logger(self.thread_logger):
                to_return = self.parent_hp_pipeline.try_run_component_in_group(trial, component_index, step, component_loader)

        self.thread_logger.writeLine(f"Ended execution of trial {trial.number}, component index {component_index}")

        return to_return
    


    def try_run_component_in_group_until_step(self, trial : optuna.Trial, component_index, final_step, component_loader : StatefulComponentLoader, should_end):

        self.check_if_should_stop_execution_earlier(trial) # check if experiment should be stopped earlier

        with override_first_logger(self.thread_logger):
            last_reported_step = self.parent_hp_pipeline.get_last_reported_step(trial, component_index)

        # IF we are to do an initial evaluation
        if last_reported_step < 0 and self.parent_hp_pipeline.do_initial_evaluation:
            with override_first_logger(self.thread_logger):
                self.parent_hp_pipeline.exec_do_initial_evaluation(trial, component_index, component_loader)
            
        if last_reported_step >= final_step:
            self.thread_logger.writeLine(f"In trial {trial.number}, component index {component_index} had already done the number of steps needed, as it did {last_reported_step} (higher or equal than {final_step})")
            
        if last_reported_step < 0:
            next_step = 1
        else:
            next_step = last_reported_step + 1

        n_steps = final_step - next_step

    
        self.thread_logger.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEPS TO DO {n_steps}, ENDS AT {n_steps} -----------------------\n")
            
        to_return = None

        for step in range(next_step, next_step + n_steps):

            if should_end[0]:
                self.thread_logger.writeLine(f"Interruped in trial {trial.number}, in component {component_index}, before step {step}, due to concurrent logic (probably some evaluation made it so the whole trial was pruned)")
                break

            self.check_if_should_stop_execution_earlier(trial)

            with override_first_logger(self.thread_logger):
                to_return = self.parent_hp_pipeline.try_run_component_in_group(trial, component_index, step, component_loader)

        self.thread_logger.writeLine(f"Ended execution of trial {trial.number}, component index {component_index}")

        return to_return

