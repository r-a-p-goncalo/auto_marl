import os
from automarl.components.basic_components.component_group import RunnableComponentGroup, setup_component_group

from automarl.components.basic_components.exec_component import State
from automarl.component import Component, ParameterSignature
from automarl.core.exceptions import common_exception_handling
from automarl.components.hp_opt.hp_optimization_pipeline import Component_to_opt_type, HyperparameterOptimizationPipeline

import math

from automarl.components.rl.evaluators.rl_std_avg_evaluator import ResultLogger
import optuna
from automarl.components.basic_components.stateful_component_loder import StatefulComponentLoader

import pandas


CONFIGURATION_PATH_OPTUNA_KEY = "configuration_path"
COMPONENT_INDEX_TO_USE_OPTUNA_KEY = "component_index_to_continue_using"


class HyperparameterOptimizationLoader(HyperparameterOptimizationPipeline):
    
    '''
    An HP optimization pipeline wich loads and unloads the components it is optimizing
    This supports algorithms which may want to return to a previous trial and continue its progress
    '''

    parameters_signature = {
                         "trainings_per_configuration" : ParameterSignature(default_value=3),
                         "use_best_component_strategy_with_index" : ParameterSignature(default_value=1),
                         "only_report_with_enough_runs" : ParameterSignature(default_value=True)
                       }
            


    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _process_input_internal(self): # this is the best method to have initialization done right after
                
        super()._process_input_internal()

        self.lg.writeLine(f"Starting processing input related to multiple components per trial in disk...")
                
        self.trial_loader_groups : dict[str, RunnableComponentGroup] = {} # the groups for each trial, each group has the same hyperparameter and different seeds

        self.trainings_per_configuration = self.get_input_value("trainings_per_configuration") # training processes per configuration, the result is the average

        self._init_component_of_trial_selection_strategy()

        self.only_report_with_enough_runs = self.get_input_value("only_report_with_enough_runs")

        if self.only_report_with_enough_runs:
            self.lg.writeLine(f"Will only report results with enough runs")

        else:
            self.lg.writeLine(f"Will reports results as they appear, not only when we have enough runs")


        self.lg.writeLine(f"Finished processing input related to multiple components per trial execution\n")


    def _init_component_of_trial_selection_strategy(self):

        self.use_best_component_strategy_with_index = self.get_input_value("use_best_component_strategy_with_index")

        if self.use_best_component_strategy_with_index >= 0:
            self.lg.writeLine(f"The strategy on using the multiple components will be to take the best of the {self.use_best_component_strategy_with_index}nth step and only continue with that")

        else:
            self.lg.writeLine(f"The strategy on using multiple components will be to train in parallel all the steps")


    def _should_report_best_component(self):
        return self.use_best_component_strategy_with_index >= 0


    def _setup_hp_results_logger(self, parameter_names): # overwrites so we also store component index when reporting results
        self.add_to_columns_of_results_logger(["experiment", "component_index", "step", *parameter_names, "result"])


    # CREATION OF COMPONENT TO OPTIMIZE -------------------------------------------------------------

    def _generate_name_for_component_group(self, trial : optuna.Trial):
        
        return f"conf_{trial.number}"


    def _create_component_to_optimize(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the component to optimize and and saver / loader for it, returning the component to optimize itself'''
        
        component_to_opt = super()._create_component_to_optimize(trial)

        group_directory = component_to_opt.get_artifact_directory()
        base_name = self._generate_name_for_component_group(trial)

        self.trial_loader_groups[trial.number] = setup_component_group(self.trainings_per_configuration, group_directory, base_name, component_to_opt, True)

        trial.set_user_attr(CONFIGURATION_PATH_OPTUNA_KEY, self.trial_loader_groups[trial.number].get_artifact_directory())

        return self.trial_loader_groups[trial.number]
        

    def _try_load_component_into_trial_loader_groups(self, trial : optuna.Trial):

        path_of_configuration = trial.user_attrs[CONFIGURATION_PATH_OPTUNA_KEY]
        
        if os.path.exists(path_of_configuration):

            self.trial_loader_groups[trial.number] = RunnableComponentGroup({
                "artifact_relative_directory" : '',
                "base_directory" : path_of_configuration,
                "create_new_directory" : False
            })

            loaders = []

            for i in range(self.trainings_per_configuration):
                loaders.append(
                    StatefulComponentLoader(
                        {"base_directory" : self.trial_loader_groups[trial.number].get_artifact_directory(),
                         "create_new_directory" : False,
                         "artifact_relative_directory" : f'{i}'
                         }
                    )
                )
                
                self.trial_loader_groups[trial.number].define_component_as_child(loaders[i])
                

            self.trial_loader_groups[trial.number].pass_input({"components_loaders_in_group" : loaders}) 

            return self.trial_loader_groups[trial.number]
        
        else:
            return None
        

    # GETING INITIALIZED COMPONENT -------------------------------------------------------
    

    def _get_loader_component_group(self, trial : optuna.Trial) -> RunnableComponentGroup:
                
        component_group : RunnableComponentGroup = self.trial_loader_groups[trial.number]
        
        return component_group


    def get_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
         
        if not trial.number in self.trial_loader_groups.keys():
            
            return self._create_component_to_optimize(trial)
        
        else: 
            return self._get_loader_component_group(trial)
        
    def load_component_to_test(self, component_loader : StatefulComponentLoader) -> Component_to_opt_type:
                
        component_to_opt = component_loader.get_component()

        self._check_running_state_of_component(component_to_opt)

        return component_to_opt

    
    def _get_component_index_of_trial(self, trial : optuna.Trial, component_index=None):

        if component_index is not None:
            return component_index
        
        else:
            to_return = trial.user_attrs.get(COMPONENT_INDEX_TO_USE_OPTUNA_KEY)

            if to_return is None:
                raise Exception(f"Trial {trial.number} does not have a component index specified and component index passed is None")
            
            self.lg.writeLine(f"Component index to use for trial {trial.number} is {to_return}")

            return to_return

    

    # NOTING RESULTS ---------------------------------------------------   



    def log_results_of_trial(self, trial : optuna.Trial, step : int, component_index : int, evaluation_results):

        result = evaluation_results["result"]

        results_to_log = {'experiment' : trial.number, "component_index" : component_index, "step" : step, **self._suggested_values_by_trials[trial.number], "result" : result}

        for key, value in results_to_log.items():
            results_to_log[key] = [value]

        self.log_results(results_to_log)  

    def get_result_for_trial_for_step(self, trial, step):

            # Count how many results we already have for this (trial, step)
            results_logger: ResultLogger = self.get_results_logger()

            df : pandas.DataFrame = results_logger.get_dataframe()

            mask = (
                (df["experiment"] == trial.number) &
                (df["step"] == step)
            )

            filtered_df = df.loc[mask]

            last_results = filtered_df["result"].tolist()

            return last_results

    def get_result_for_component_index_for_step(self, trial, component_index, step):

            # Count how many results we already have for this (trial, step)
            results_logger: ResultLogger = self.get_results_logger()

            df : pandas.DataFrame = results_logger.get_dataframe()

            mask = (
                (df["experiment"] == trial.number) &
                (df["component_index"] == component_index) &
                (df["step"] == step)
            )

            filtered_df = df.loc[mask]

            last_results = filtered_df["result"].tolist()

            return last_results


    def load_and_report_results(self, trial : optuna.Trial, component_index : int, step : int, evaluation_results):

        self.log_results_of_trial(trial, step, component_index, evaluation_results)

        last_results = self.get_result_for_trial_for_step(trial, step)

        n_results_for_step = len(last_results)

        # we have enough results to report if we have the number of results for all trainings or if we are above the treshold where the best component is defined
        enough_runs = n_results_for_step == self.trainings_per_configuration or (self.use_best_component_strategy_with_index > 0 and step > self.use_best_component_strategy_with_index)

        if self.only_report_with_enough_runs and not enough_runs:
            self.lg.writeLine(f"Not enough configurations to report, current is {n_results_for_step}, needed is {self.trainings_per_configuration}. Current results are {last_results}")

        else:
            
            # if we are to use the best component, we always report the best
            if self.use_best_component_strategy_with_index > 0:

                result_to_report = max(last_results)
                self.lg.writeLine(
                    f"Reporting result {result_to_report}, maximum of results: {last_results}"
                )

            else:

                result_to_report = sum(last_results) / len(last_results) if len(last_results) > 1 else last_results[0]
                self.lg.writeLine(f"Reporting result {result_to_report}, average of results: {last_results}")

            self.report_value_for_optuna(trial, result_to_report, step)
        
        return enough_runs


    def get_last_reported_step(self, trial : optuna.Trial, component_index=None):

            
            results_logger: ResultLogger = self.get_results_logger()

            df : pandas.DataFrame = results_logger.get_dataframe()

            if component_index is not None:
                mask = (
                    (df["experiment"] == trial.number) & 
                    (df["component_index"] == component_index)
                )
            
            else:
                mask = (
                    (df["experiment"] == trial.number)
                )
            

            filtered_df = df.loc[mask]

            if len(filtered_df) == 0:
                return -1

            last_step = filtered_df["step"].max()

            return last_step


    def _compute_result_for_trial_run(self, trial : optuna.Trial):

        '''Computes the result for a trial that was already run, to be used inside the objective'''

        self.lg.writeLine(f"Computing results for trial {trial.number}")
    
        df_for_this_trial = self.get_results_dataframe_for_trial(trial)

        max_step = df_for_this_trial["step"].max()

        last_step_for_this_trial_rows = df_for_this_trial[df_for_this_trial["step"] == max_step]

        if self.use_best_component_strategy_with_index > 0:
            
            # Find row with maximum result
            idx_max = last_step_for_this_trial_rows["result"].idxmax()
            best_row = last_step_for_this_trial_rows.loc[idx_max]

            component_index_with_maximum_result = int(best_row["component_index"])
            best_result = best_row["result"]

            if self.use_best_component_strategy_with_index > 0 and max_step >= self.use_best_component_strategy_with_index:
                # Store component index so we can continue using it
                trial.set_user_attr(COMPONENT_INDEX_TO_USE_OPTUNA_KEY, component_index_with_maximum_result)

            return best_result

        else:

            n_results_for_step = len(last_step_for_this_trial_rows)

            last_results = last_step_for_this_trial_rows["result"].tolist()

            if n_results_for_step != self.trainings_per_configuration:
                raise Exception(f"Mismatch between number of trainings that should have completed for trial {trial.number} ({self.trainings_per_configuration}) and actual number: {n_results_for_step}")

            return sum(last_results) / len(last_results) 
        

    # RUNNING OPTIMIZATION ---------------------------------------------------------------



    def exec_do_initial_evaluation(self, trial : optuna.Trial, component_index, component_loader : StatefulComponentLoader):

        '''Does initial evaluation of a single component'''

        self.lg.writeLine(f"Doing initial evaluation of trial {trial.number} before initiating training...")
            
        component_to_test_path = str(component_loader.get_artifact_directory())

        try:
                component_to_test = self.load_component_to_test(component_loader)

        except Exception as e:
                self.on_general_exception_trial(e, component_to_test_path, trial, component_index)

        evaluation_results = self._try_evaluate_component(component_to_test_path, trial, component_to_test)

        self.load_and_report_results(trial, component_index, 0, evaluation_results)

        self.lg.writeLine(f"Results were: {evaluation_results}\n")

        del component_to_test
        component_loader.save_and_onload_component()

        self.check_if_should_stop_execution_earlier(trial) # check if experiment should be stopped earlier

    
    def _try_run_single_component_in_group(self, trial : optuna.Trial, component_index, component_loader : StatefulComponentLoader):
        
        '''Runs a single component, using a detached strategy (creates a separate process for it)'''

        self.check_if_should_stop_execution_earlier(trial)

        component_to_test_path = str(component_loader.get_artifact_directory())

        self.load_component_to_test(component_loader)

        component_to_test_state = component_loader.get_loaded_component_state()

        if State.equals_value(component_to_test_state, State.OVER):
            self.lg.writeLine(f"In trial {trial.number}, component index {component_index} was already over, will not execute it again")
            component_loader.unload_component_if_loaded_with_retries(lg=self.lg)
            return

        try:
  
            component_loader.save_component()
            component_loader.unload_component_if_loaded_with_retries(lg=self.lg)

            component_loader.detach_run_component(
                                                        to_wait=True, 
                                                        global_logger_level='INFO'
                                                        )
            
            self.load_component_to_test(component_loader)
            component_loader.unload_component_if_loaded_with_retries(lg=self.lg)


        except Exception as e:
            self.on_exception_running_trial(e, component_to_test_path, trial, component_index)



    def try_run_component_in_group(self, trial : optuna.Trial, component_index, step, component_loader : StatefulComponentLoader):

        '''Does an execution step of the objective, running and evaluating a component'''

        self.lg.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEP {step} -----------------------\n")

        component_to_test_path = str(component_loader.get_artifact_directory())

        self.lg.writeLine(f"Component is in {component_to_test_path}, loading and running it...\n")

        try:

                self._try_run_single_component_in_group(trial, component_index, component_loader)

                try:
                        component_to_test = self.load_component_to_test(component_loader)

                except Exception as e:
                    self.on_general_exception_trial(e, component_to_test_path, trial, component_index)

                self.lg.writeLine("Finished running component, evaluating it...\n")

                evaluation_results = self._try_evaluate_component(component_to_test_path, trial, component_to_test)

                # we report and evaluate results
                try:

                    self.lg.writeLine(f"Finished evaluating component, reported results were {evaluation_results}\n")

                    enough_runs = self.load_and_report_results(trial, component_index, step, evaluation_results)

                    self.lg.writeLine(f"Ended step {step} for component {component_index} in trial {trial.number}\n") 

                    if enough_runs:
                        self.check_if_should_prune_trial(trial, step)

                    return evaluation_results
                
                except Exception as e:

                    if isinstance(e, optuna.TrialPruned):
                        raise e
                    
                    else:
                        self.on_general_exception_trial(e, component_to_test_path, trial, component_index)

        except Exception as e:

                if not isinstance(e, optuna.TrialPruned):
                    self.lg.writeLine(f"Error in trial {trial.number} in component (index) {component_index} at step {step}: {e}")

                raise e
        

            
    def try_run_component_in_group_all_steps(self, trial : optuna.Trial, component_index, n_steps, component_loader : StatefulComponentLoader, should_end):

        '''Runs a component in all steps'''

        self.check_if_should_stop_execution_earlier(trial) # check if experiment should be stopped earlier

        last_reported_step = self.get_last_reported_step(trial, component_index)

        # IF we are to do an initial evaluation
        if last_reported_step < 0 and self.do_initial_evaluation:
            self.exec_do_initial_evaluation(trial, component_index, component_loader)
            

        if last_reported_step < 0:
            next_step = 1
        else:
            next_step = last_reported_step + 1

        self.lg.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEPS TO DO {n_steps} -----------------------\n")
            
        to_return = None

        for step in range(next_step, next_step + n_steps):
            to_return = self.try_run_component_in_group(trial, component_index, step, component_loader)


        self.lg.writeLine(f"Ended execution of trial {trial.number}, component index {component_index}")

        return to_return
    


    def try_run_component_in_group_until_step(self, trial : optuna.Trial, component_index, final_step, component_loader : StatefulComponentLoader, should_end):

        self.check_if_should_stop_execution_earlier(trial) # check if experiment should be stopped earlier

        last_reported_step = self.get_last_reported_step(trial, component_index)

        # IF we are to do an initial evaluation
        if last_reported_step < 0 and self.do_initial_evaluation:
            self.exec_do_initial_evaluation(trial, component_index, component_loader)
            
        if last_reported_step >= final_step:
            self.lg.writeLine(f"In trial {trial.number}, component index {component_index} had already done the number of steps needed, as it did {last_reported_step} (higher or equal than {final_step})")
            
        if last_reported_step < 0:
            next_step = 1
        else:
            next_step = last_reported_step + 1

        n_steps = final_step - next_step + 1

    
        self.lg.writeLine(f"----------------------- TRIAL: {trial.number}, COMPONENT_INDEX: {component_index}, STEPS TO DO {n_steps}, ENDS AT {n_steps} -----------------------\n")
            
        to_return = None

        for step in range(next_step, final_step + 1):
            to_return = self.try_run_component_in_group(trial, component_index, step, component_loader)

        self.lg.writeLine(f"Ended execution of trial {trial.number}, component index {component_index}")

        return to_return


    def _run_optimization(self, trial: optuna.Trial):

        '''Runs the optimization function for a trial (essentially the objective)'''

        if self.check_if_should_stop_execution_earlier(raise_exception=False):
            self.check_if_should_stop_execution_earlier()

        self.lg.writeLine(f"----------------------- TRIAL {trial.number} --------------------------------------")

        self.lg.writeLine(f"Generating component if it does not exist...")

        # this is to force create the component for the trial
        loader : RunnableComponentGroup = self.get_component_to_test(trial)
        loader.unload_all_components_with_retries(lg=self.lg)

        self.lg.writeLine(f"Component group was created")

        for step in range(1, self.n_steps + 1):

            best_index = trial.user_attrs.get(COMPONENT_INDEX_TO_USE_OPTUNA_KEY)

            if best_index is None:
                for component_index in range(self.trainings_per_configuration):
                    self.try_run_component_in_group(trial, component_index, step, loader.get_loader(component_index))

            else:
                self.lg.writeLine(f"On step {step}, noting that best (and only) component index to use is {best_index}")
                self.try_run_component_in_group(trial, best_index, step, loader.get_loader(best_index))


        return self._compute_result_for_trial_run(trial)
    

    def _try_load_all_resumed_trials(self, trials : list[optuna.trial.FrozenTrial]):

        self.lg.writeLine(f"Trying to load loaders of {len(trials)} queued trials...")

        for trial in trials:
            if self._try_load_component_into_trial_loader_groups(trial) is not None:
                self.lg.writeLine(f"Loaded trial {trial.number} from disk")
            
            else:
                self.lg.writeLine(f"Could not find trial {trial.number} in disk")
      

    # RUNNING A TRIAL -----------------------------------------------------------------------
    


    def after_trial(self, study : optuna.Study, trial : optuna.trial.FrozenTrial):
        
        '''
        Called when a trial is over
        It is passed to optuna in the callbacks when the objective is defined
        '''        

        super().after_trial(study, trial)
                        
        component_group = self._get_loader_component_group(trial)
        component_group.unload_all_components_with_retries()

      
    



    # DEALING WITH EXCEPTIONS -------------------------------------------------------------------

    def on_exception_evaluating_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial, component_index=None):

        component_index_str = f" in component index {component_index}" if component_index is not None else ""

        self.lg.writeLine(f"ERROR: CAN'T EVALUATE TRIAL {trial.number}{component_index_str}")

        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_specific_path = "on_evaluate_error_report.txt"
        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.lg.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.lg, exception, error_report_path)

        raise exception
    
    

    def on_exception_running_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial, component_index):

        '''Deals with an exception while running a trial and re-raises it'''

        self.lg.writeLine(f"ERROR RUNNING TRIAL {trial.number} in component index {component_index}")

        error_report_specific_path = "on_run_error_report.txt"
        
        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.lg.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.lg, exception, error_report_path)

        raise exception
    

        
    def on_general_exception_trial(self, exception : Exception, component_to_test_path, trial : optuna.Trial, component_index=None, reraise_exception=True):

        '''Deals with a general exception with no specified strategy and re-raises it'''

        component_index_str = f" in component index {component_index}" if component_index is not None else ""

        self.lg.writeLine(f"ERROR IN TRIAL {trial.number}{component_index_str}")

        error_report_specific_path = "general_exception.txt"
        
        component_to_test_path = os.path.abspath(component_to_test_path)

        error_report_path = os.path.join(component_to_test_path, error_report_specific_path)

        self.lg.writeLine(f"Storing error report in configuration, path {error_report_path}\nError: {exception}")

        common_exception_handling(self.lg, exception, error_report_path)

        if reraise_exception:
            raise exception
    


    # EARLIER INTERRUPTION OF TRAINING ---------------------------------------------------

    def _check_running_state_of_component(self, component_to_opt : Component):
    
        '''Checks running state of a loaded component (not component group), raising an exception if it was an error'''

        running_state = component_to_opt.values.get("running_state", None)

        if running_state is not None:

            if State.equals_value(running_state, State.ERROR):
                raise Exception(f"Error when component in {component_to_opt.get_artifact_directory()} was got") 
        

    def _check_if_should_stop_execution_earlier(self):

        '''
        This allows for trials to receive a stop signal in the form of a file with the name "__stop"
        '''

        should_stop = super()._check_if_should_stop_execution_earlier()

        if should_stop:
            return True
        
        else:
            stop_path = os.path.join(self.get_artifact_directory(), "__stop")
            return os.path.exists(stop_path)
                
    
    def _on_earlier_interruption(self):

        '''Verifies if the __stop file exists and removes it if so'''

        super()._on_earlier_interruption()

        self.lg.writeLine(f"Hyperparamter optimization process was interrupted")
        
        stop_path = os.path.join(self.get_artifact_directory(), "__stop")

        if os.path.exists(stop_path):
            try:
                os.remove(stop_path)
                self.lg.writeLine(f"Stop sign was succecsfuly removed")

            except FileNotFoundError:
                self.lg.writeLine(f"Tried deleting stop signal, but was not there")
                
            except OSError as e:
                self.lg.writeLine(f"WARNING: failed to remove stop signal at {stop_path}: {e}")


    

                


                
    