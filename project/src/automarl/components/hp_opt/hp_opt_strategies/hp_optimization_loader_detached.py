from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
import threading
import time
from automarl.components.basic_components.component_group import RunnableComponentGroup

from automarl.component import ParameterSignature
from automarl.components.hp_opt.hp_opt_strategies.hp_optimization_loader import COMPONENT_INDEX_TO_USE_OPTUNA_KEY, CONFIGURATION_PATH_OPTUNA_KEY, HyperparameterOptimizationLoader
from automarl.components.hp_opt.hp_opt_strategies.workers.hp_worker import HyperparameterOptimizationWorkerIndexed
from automarl.components.hp_opt.hp_optimization_pipeline import Component_to_opt_type

import math

import optuna

from automarl.components.loggers.logger_component import override_first_logger, use_logger

from automarl.components.basic_components.exec_component import StopExperiment
from automarl.components.loggers.global_logger import globalWriteLine

MINIMUM_SLEEP = 1
SLEEP_INCR_RATE = 2
MAX_SLEEP = 60

class HyperparameterOptimizationPipelineLoaderDetached(HyperparameterOptimizationLoader):
    
    '''
    An HP optimization pipeline wich loads and unloads the components it is optimizing
    This supports algorithms which may want to return to a previous trial and continue its progress
    '''

    parameters_signature = {
                         "trainings_at_a_time" : ParameterSignature(default_value=6),
                       }
            


    
    # INITIALIZATION -----------------------------------------------------------------------------


    def _process_input_internal(self): # this is the best method to have initialization done right after
                
        super()._process_input_internal()
                
        self.trainings_at_a_time = self.get_input_value("trainings_at_a_time") # training processes that can be done at a time

        # trainings wich we should prioritize running at the same time (they are the minimum trainings we'll be doing even if we try our best to minimize it)
        # we track this to distribute best our resources by this number of configurations, and give only sparse resource to other configurations
        # the priority should be to give resources to trials with the lowest index
        self.beneficted_trainings = math.ceil(self.trainings_at_a_time / self.trainings_per_configuration)

        self.lg.writeLine(f"For {self.trainings_per_configuration} trainings per configuration, with {self.trainings_at_a_time} trainings being done at a time, the minimum number of hyperparameter configurations being tested at a time should be {self.beneficted_trainings}")

        # semaphore to use the evaluator
        self.evaluator_sem = threading.RLock()  

        # a list of [trial_number, sem, configurations_not_attributed_to_worker], to use in prioritizing runn
        self.trainings_remaining_per_thread_sem = threading.RLock()    
        self.trainings_remaining_per_thread = [None] * self.trainings_at_a_time

        self.optuna_usage_sem = threading.RLock()  


        self._setup_workers()

        self.results_logger_sem = threading.RLock()  

        self.lg.writeLine(f"Finished processing input related to multiple component, detached execution")




    # THREADS SETUP ---------------------------------------------------

    def _setup_workers(self):

        self.workers : list[HyperparameterOptimizationWorkerIndexed] = []

        self.lg.writeLine(f"Total workers: {self.trainings_at_a_time}")

        for thread_index in range(self.trainings_at_a_time):

            worker = HyperparameterOptimizationWorkerIndexed(self, thread_index)

            self.workers.append(worker)

            self.lg.writeLine(f"Created worker {worker.thread_index}, its logging will be made at {worker.thread_logger.get_artifact_directory()}")


        self.lg.writeLine()


    def report_value_for_optuna(self, trial : optuna.Trial, value, step):
        with self.optuna_usage_sem:
            super().report_value_for_optuna(trial, value, step)

    
    def _try_evaluate_component(self, component_to_test_path, trial : optuna.Trial, component_to_test = None) -> float:

        # this is only to guarantee only one results is evaluated at the same time
        with self.evaluator_sem:
            return super()._try_evaluate_component(component_to_test_path, trial, component_to_test)

    def _setup_hp_results_logger(self, parameter_names):
        self.add_to_columns_of_results_logger(["experiment", "component_index", "step", *parameter_names, "result"])


    def log_results_of_trial(self, trial : optuna.Trial, step : int, component_index : int, evaluation_results):

        with self.results_logger_sem:
            return super().log_results_of_trial(trial, step, component_index, evaluation_results)

    # OPTIMIZATION -------------------------------------------------------------------------


    def _create_component_to_optimize(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        '''Creates the component to optimize and and saver / loader for it, returning the component to optimize itself'''
        
        with self.optuna_usage_sem:
            return super()._create_component_to_optimize(trial)

    
    # RUNNING THE WORKERS ------------------------------------------------------

    def _run_available_worker(self, trial : optuna.Trial, component_index, n_steps, should_end, index_in_trainings_remaining):

        '''Waits until it can aquire a worker and then runs a job with it, freeing it in the end'''

        worker = self._aquire_available_worker()

        to_return = self._run_worker(worker, trial, component_index, 
                                     n_steps, should_end, index_in_trainings_remaining)

        return to_return
    
    
    def _run_available_worker_till_step(self, trial : optuna.Trial, component_index, last_step, should_end, index_in_trainings_remaining):

        '''Waits until it can aquire a worker and then runs a job with it, freeing it in the end'''

        #self.lg.writeLine(f"HERE IN _run_available_worker_till_step: {trial.number}")

        worker = self._aquire_available_worker()

        to_return = self._run_worker_till_step(worker, trial, component_index, last_step, should_end, index_in_trainings_remaining)

        return to_return
    
    def _verify_and_aquire_worker_not_running_on_full_configuration(self, worker : HyperparameterOptimizationWorkerIndexed, trial, list_where_worker_working):

        '''
        Aquires a configuration to be trained in a trial for a worker
        '''

        [_, sem, current_number] = list_where_worker_working

        with sem:
            
            if current_number <= 0:
                worker.free_worker()
                raise Exception(f"Worker {worker.thread_index} attributed to trial {trial.number}, when that trial had already all its configurations trained / being trained (missing {current_number} trainings)")
            
            list_where_worker_working[2] = current_number - 1

    def _let_go_worker_list(self, worker, trial, list_where_worker_working):

        [_, sem, current_number] = list_where_worker_working

        with sem:
           list_where_worker_working[2] = current_number + 1


    def _run_worker(self, worker : HyperparameterOptimizationWorkerIndexed , trial : optuna.Trial, component_index, n_steps, should_end, index_in_trainings_remaining):

            '''Runs a job with a worker and then frees it'''

            self._verify_and_aquire_worker_not_running_on_full_configuration(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])

            with override_first_logger(worker.thread_logger):
                component_index = self._get_component_index_of_trial(trial, component_index)

            try:
                to_return = worker.try_run_component_in_group_all_steps(trial, component_index, n_steps, 
                                                    self._get_loader_component_group(trial).get_loader(component_index), 
                                                    should_end)

            except Exception as e:
                should_end[0] = True # if we receive an exception, we flag that the training for this trial should end
                self._let_go_worker_list(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])
                worker.free_worker()
                raise e

            self._let_go_worker_list(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])
            worker.free_worker()
            return to_return
    

    
    def _run_worker_till_step(self, worker : HyperparameterOptimizationWorkerIndexed , trial : optuna.Trial, component_index, last_step, should_end, index_in_trainings_remaining):

            '''Runs a job with a worker and then frees it'''

            self._verify_and_aquire_worker_not_running_on_full_configuration(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])

            with override_first_logger(worker.thread_logger):
                component_index = self._get_component_index_of_trial(trial, component_index)

            try:
                to_return = worker.try_run_component_in_group_until_step(trial, component_index, last_step, self._get_loader_component_group(trial).get_loader(component_index), should_end)

            except Exception as e:
                should_end[0] = True # if we receive an exception, we flag that the training for this trial should end
                self._let_go_worker_list(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])
                worker.free_worker()
                raise e

            self._let_go_worker_list(worker, trial, self.trainings_remaining_per_thread[index_in_trainings_remaining])
            worker.free_worker()
            return to_return
    


    def _aquire_available_worker(self):

        '''Aquires the next available worker, waiting if necessary'''

        next_worker_index = 0

        sleep_time = MINIMUM_SLEEP

        while True:

            self.check_if_should_stop_execution_earlier()

            next_worker = self.workers[next_worker_index]
            worker = next_worker.aquire_if_not_busy()

            if worker is not None:
                return worker

            next_worker_index = next_worker_index + 1

            if next_worker_index >= len(self.workers): # we did a full search, we wait before doing it again
                next_worker_index = 0
                time.sleep(sleep_time)
                sleep_time = min(sleep_time * SLEEP_INCR_RATE, MAX_SLEEP) # we increase the sleep time

            
    def _get_next_available_index_trainings(self):
        
        '''
        Gets the next available index to represent a trial job
        Each trial job will wait for available workers
        '''

        while True:

            self.check_if_should_stop_execution_earlier()

            with self.trainings_remaining_per_thread_sem:
                
                for i in range(len(self.trainings_remaining_per_thread)):

                    if self.trainings_remaining_per_thread[i] is None:
                    
                        self.trainings_remaining_per_thread[i] = 0 # we change its value so no other search will find this index
                        return i
                    
            time.sleep(MINIMUM_SLEEP)

    
    def _wait_for_previous_trials_to_be_occupied(self, trial : optuna.Trial):

        '''If a trial should wait before trying to acquire workers, waits'''

        if trial.number < self.beneficted_trainings:
            return
        
        to_wait = True

        while to_wait:

            self.check_if_should_stop_execution_earlier()

            trials_to_receive_resources_before_this_one = 0
            to_wait = False
            
            with self.trainings_remaining_per_thread_sem:

                for i, value in enumerate(self.trainings_remaining_per_thread):

                    if value is not None and value != 0:
                        [trial_number, sem, jobs_to_go] = value

                        if trial_number < trial.number: # if there is a trial previous of this one

                            with sem:
                                jobs_to_go = self.trainings_remaining_per_thread[i][2]

                                if jobs_to_go >= 1: # and it has a configuration that is not yet being trained

                                    trials_to_receive_resources_before_this_one += 1

                                    # if the number of trials that are ahead of this one on wanting to aquire a resource is higher than the number of beneficted trainings
                                    if trials_to_receive_resources_before_this_one >= self.beneficted_trainings:
                                        to_wait = True
                                        break

            if to_wait:
                time.sleep(MAX_SLEEP) # we wait   


    # CONCURRENT TRIAL LOGIC ---------------------------------------------------      

    def _compute_result_for_trial_run(self, trial):
        with self.results_logger_sem:
            return super()._compute_result_for_trial_run(trial)
        
    def _generate_futures_for_single_trial_with_normal_strategy(self, trial: optuna.Trial, index_in_trainings_remaining, first_worker : HyperparameterOptimizationWorkerIndexed, 
                                           run_first_worker_fun, run_other_workers_fun, should_end):
        
            list_of_futures = []
            futures = []

            futures.append(
                [
                    run_first_worker_fun,
                    first_worker,
                    trial,
                    0,
                    self.n_steps,
                    should_end,
                    index_in_trainings_remaining
                ]
            )

            # remaining jobs go through dynamic worker acquisition
            for component_index in range(1, self.trainings_per_configuration):
                futures.append(
                    [
                        run_other_workers_fun,
                        trial,
                        component_index,
                        self.n_steps,
                        should_end,
                        index_in_trainings_remaining
                    ]
                )

            list_of_futures.append(futures)

            return list_of_futures


        
    def _generate_futures_for_single_trial_with_best_strategy(self, trial: optuna.Trial, index_in_trainings_remaining, first_worker : HyperparameterOptimizationWorkerIndexed, 
                                           run_first_worker_fun, run_other_workers_fun, should_end):
    
            first_worker.thread_logger.writeLine(f"Trial {trial.number} does not have registered component index as the only one to be used")
            
            last_step_trial_run = self.get_last_reported_step(trial)
            last_step_trial_run = 0 if last_step_trial_run < 0 else last_step_trial_run

            step_to_end_this_execution = last_step_trial_run + self.n_steps
            
            first_worker.thread_logger.writeLine(f"Last step for {trial.number} was {last_step_trial_run}, and this execution with {self.n_steps} steps will end at step {step_to_end_this_execution}")

            if step_to_end_this_execution < self.use_best_component_strategy_with_index:
                first_worker.thread_logger.writeLine(f"As the it is bellow the range to choose the best component (step {self.use_best_component_strategy_with_index}), will use normal run strategy")

                return self._generate_futures_for_single_trial_with_normal_strategy(
                trial,
                index_in_trainings_remaining,
                first_worker,
                run_first_worker_fun,
                run_other_workers_fun,
                should_end
            )

            else:
                steps_until_decision = self.use_best_component_strategy_with_index - last_step_trial_run
                first_worker.thread_logger.writeLine(f"There are {steps_until_decision} steps until decision of best component")

            list_of_futures = []

            futures = []

            futures.append(
                [
                    run_first_worker_fun,
                    first_worker, # worker
                    trial, # trial
                    0, # component index
                    steps_until_decision, # steps to run
                    should_end,
                    index_in_trainings_remaining
                ]
            )

            # remaining jobs go through dynamic worker acquisition
            for component_index in range(1, self.trainings_per_configuration):
                futures.append(
                    [
                        run_other_workers_fun,
                        trial,
                        component_index,
                        steps_until_decision,
                        should_end,
                        index_in_trainings_remaining
                    ]
                )

            list_of_futures.append(futures)

            if self.n_steps > steps_until_decision:
                
                list_of_futures.append(
                    [
                        [
                            self._compute_result_for_trial_run, # this is so the best step is registered
                            trial
                        ]
                    ]
                )

                list_of_futures.append(
                    [
                        [
                            run_other_workers_fun,
                            trial,
                            None,
                            self.n_steps - steps_until_decision,
                            should_end,
                            index_in_trainings_remaining
                        ]
                    ]
                )

            return list_of_futures


    def _generate_futures_for_single_trial(self, trial: optuna.Trial, index_in_trainings_remaining, first_worker : HyperparameterOptimizationWorkerIndexed, 
                                           run_first_worker_fun, run_other_workers_fun, should_end):
    
        '''Generated a list of lists of futures for a single trial. Between lists of futures, waiting is expected'''

        list_of_futures = []

        component_index_to_use = trial.user_attrs.get(COMPONENT_INDEX_TO_USE_OPTUNA_KEY)

        if component_index_to_use is not None:
            
            futures = []

            first_worker.thread_logger.writeLine(f"Trial {trial.number} has registered component index {component_index_to_use} as the only one to be used")

            futures.append(
                [
                    run_first_worker_fun,
                    first_worker,
                    trial,
                    component_index_to_use,
                    self.n_steps,
                    should_end,
                    index_in_trainings_remaining
                ]
            )

            list_of_futures.append(futures)

        elif self.use_best_component_strategy_with_index > 0: # we are using the best component strategy, with the component being ran for the first time
            
            list_of_futures = [*list_of_futures, *self._generate_futures_for_single_trial_with_best_strategy(
                trial,
                index_in_trainings_remaining,
                first_worker,
                run_first_worker_fun,
                run_other_workers_fun,
                should_end
            )]


        else:
            list_of_futures = [*list_of_futures, *self._generate_futures_for_single_trial_with_normal_strategy(
                trial,
                index_in_trainings_remaining,
                first_worker,
                run_first_worker_fun,
                run_other_workers_fun,
                should_end
            )]


        return list_of_futures
    
    
    def _execute_futures_of_single_trial(self, trial : optuna.Trial, futures, executor : ThreadPoolExecutor, should_end):

            exceptions_to_raise = []

            for i in range(len(futures)):
                if isinstance(futures[i], list):
                    futures[i] = executor.submit(*futures[i])

            # wait for all jobs
            for future in futures:

                if future.cancelled():
                    continue

                exc = None

                try:
                    future.result()
                
                except Exception as e:
                    exc = e

                if exc is not None:

                    should_end[0] = True

                    # by interrupting, exception or prunning, this is always the shutdown strategy
                    self._cancel_futures(futures)

                    exceptions_to_raise.append(exc)

                elif should_end[0]:

                    self._cancel_futures(futures)

            if len(exceptions_to_raise) == 0: # if there were no errors
                executor.shutdown(wait=False)

            self._raise_exception_in_execution(exceptions_to_raise)



    def _run_concurrent_experiments_for_single_trial(self, trial: optuna.Trial, index_in_trainings_remaining, run_till_step=False):
        
        # we only advance when we manage to get a non busy worker
        first_worker = self._aquire_available_worker()

        if run_till_step:
            run_first_worker_fun = self._run_worker_till_step
            run_other_workers_fun = self._run_available_worker_till_step

        else:
            run_first_worker_fun = self._run_worker
            run_other_workers_fun = self._run_available_worker

        first_worker.thread_logger.writeLine(f"------------------- TRIAL {trial.number} --------------------------")

        with use_logger(first_worker.thread_logger):
            
            # this is to force create the component for the trial
            loader : RunnableComponentGroup = self.get_component_to_test(trial)
            loader.unload_all_components_with_retries(lg=first_worker.thread_logger)
            del loader

        should_end = [False]

        
        list_of_futures = self._generate_futures_for_single_trial(trial, index_in_trainings_remaining, first_worker,
                                                          run_first_worker_fun, run_other_workers_fun, should_end)

        for futures in list_of_futures:
            executor = ThreadPoolExecutor(max_workers=self.trainings_at_a_time)
            self._execute_futures_of_single_trial(trial, futures, executor, should_end)


    def _acquire_index_in_trainings_remaining(self, trial: optuna.Trial):

        index_in_trainings_remaining = self._get_next_available_index_trainings() # we get an index to represent this trial (in theory, there should always be one available)
        
        with self.trainings_remaining_per_thread_sem:
            self.trainings_remaining_per_thread[index_in_trainings_remaining] = [trial.number, threading.RLock(), self.trainings_per_configuration]

        return index_in_trainings_remaining
    
    
    def _let_go_of_index_in_trainings_remaining(self, trial : optuna.Trial, index_in_trainings_remaining):
    
            with self.trainings_remaining_per_thread_sem:
                self.trainings_remaining_per_thread[index_in_trainings_remaining] = None # we let go of the index



    def _run_optimization(self, trial: optuna.Trial, trial_run_function=None, trial_run_funs_params={}):

        '''Runs the optimization function for a trial (essentially the objective)'''

        if self.check_if_should_stop_execution_earlier(raise_exception=False):
            self.check_if_should_stop_execution_earlier()
    
        if trial_run_function is None:
            trial_run_function = self._run_concurrent_experiments_for_single_trial

        if trial.number > self.beneficted_trainings and trial.number < self.trainings_at_a_time:
            time.sleep(60) # the idea is to do the maximum number of works per trial before going to the next trial
    
        index_in_trainings_remaining = self._acquire_index_in_trainings_remaining(trial)

        try:
            trial_run_function(trial, index_in_trainings_remaining, **trial_run_funs_params)
            self._let_go_of_index_in_trainings_remaining(trial, index_in_trainings_remaining)

        except Exception as e:

            with self.trainings_remaining_per_thread_sem:
                self.trainings_remaining_per_thread[index_in_trainings_remaining] = None

            raise e
        
        to_return = self._compute_result_for_trial_run(trial)

        return to_return
        
        
    def _try_resume_single_trial(self, trial : optuna.trial.FrozenTrial):

        '''Tries resuming a single trial'''

        try:
            value = self._run_optimization(trial, self._run_concurrent_experiments_for_single_trial, {"run_till_step" : True})
            return trial, value, None
        
        except Exception as e:
            return trial, None, e


    def _try_resuming_unfinished_trials(self, trials : list[optuna.trial.FrozenTrial]):

        super()._try_resuming_unfinished_trials(trials)

        self._try_load_all_resumed_trials(trials)

        self.run_trials(trials, running_method=self._try_resume_single_trial)


    def _raise_exception_in_execution(self, exceptions_to_raise, raise_exception=True):

        '''Done to check which exceptions were noted in execution of a single trial and raise experiments with priority'''

        if len(exceptions_to_raise) > 0:

            for exception_to_raise in exceptions_to_raise:
                if not isinstance(exception_to_raise, (StopExperiment, optuna.TrialPruned)):
                    if raise_exception:    
                        raise exception_to_raise
                    else:
                        return exception_to_raise

            for exception_to_raise in exceptions_to_raise:
                if not isinstance(exception_to_raise, StopExperiment):
                    if raise_exception:
                        raise exception_to_raise
                    else:
                        return exception_to_raise
            
            if raise_exception:
                raise exceptions_to_raise[0]
            else:
                return exception_to_raise        
        

    def _generate_futures_for_trials(self, executor : ThreadPoolExecutor, trials, running_method):
        futures = []

        if isinstance(trials, list):

            for trial in trials:
                futures.append(executor.submit(running_method, trial))

        elif isinstance(trials, int):
            for _ in range(trials):
                futures.append(executor.submit(running_method))

        else:
            raise Exception(f"Trials must be a list of trials or an integer specifying the number of trials")

        return futures
    

    def _cancel_futures(self, futures):
        '''
        This is a workaround for when as_completed and executor.shutdown(wait=False, cancel_futures=True) is not working properly

        Iterate over the futures with if future.cancelled(): pass with this
        '''
        for f in futures: f.cancel()

    def run_trials(self, trials, running_method=None, steps_to_run=None, mark_trials_as_completed=True):

        '''
        
        @returns results of all trials, results of non pruned trials, non pruned trials
        '''
        
        if running_method is None:
            running_method = self._run_single_trial

        self.lg.writeLine(f"Running {len(trials) if isinstance(trials, list) else trials} trials with running method: {running_method.__name__}, steps to run: {self.n_steps if steps_to_run is None else steps_to_run}, mark trials as completed: {mark_trials_as_completed}")

        results = [] # all results
        completed_results = [] # results of trials that were completed or at least not interrupted
        
        executor = ThreadPoolExecutor(max_workers=self.trainings_at_a_time)

        if steps_to_run is not None:
            old_n_steps = self.n_steps
            self.n_steps = steps_to_run

        futures = self._generate_futures_for_trials(executor, trials, running_method)

        exceptions_to_raise = [] # we delay the raising of an exception as to allow results to be properly dealt with

        for future in futures: # as_completed was not working properly with cancelled futures

            if future.cancelled(): 
                continue

            try:
                trial, value, exception = future.result()

            except Exception as e:
                exception = e

            if exception is not None:

                if isinstance(exception, optuna.TrialPruned):
                    self.mark_trial_as_pruned(trial)

                elif isinstance(exception, StopExperiment):

                    exceptions_to_raise.append(exception)
                    self.lg.writeLine(f"Trial {trial.number} over due to signal to stop the experiment")
                    self._cancel_futures(futures)
                    
                else:
                    self.lg.writeLine(f"REACHED EXCEPTION NOTICE IN RUN TRIALS FOR TRIAL {trial.number}")
                    self.mark_trial_as_failed(trial)
                    
                    trial_component_path = trial.user_attrs.get(CONFIGURATION_PATH_OPTUNA_KEY)
                    if trial_component_path is not None:
                        self.on_general_exception_trial(exception, trial_component_path, trial, reraise_exception=False)

                    if not self.continue_after_error:
                        self.lg.writeLine(f"There was an error in trial {trial.number}, as <continue_after_error> is set to false, will cancel other trials, error was: {exception}")
                        self.stop_execution_earlier()
                        self._cancel_futures(futures)
                        exceptions_to_raise.append(exception)

                    else:
                        self.lg.writeLine(f"Error in trial {trial.number}, as <continue_after_error> is set to true, will not cancel other trials, error was: {exception}")
                        self.values["trials_done_in_this_execution"] += 1 # we only count as a completed trial if it did not end the test
            

            else:
                completed_results.append((trial, value))

                if mark_trials_as_completed:
                    self.mark_trial_as_complete(trial, value)

            results.append((trial, value))

        if len(exceptions_to_raise) == 0:
            executor.shutdown(wait=True)

        if steps_to_run is not None:
            self.n_steps = old_n_steps

        self._raise_exception_in_execution(exceptions_to_raise)

        return results, completed_results
        

    # CHANGING TRIALS -------------

    def mark_trial_as_complete(self, trial, value):
        with self.optuna_usage_sem:
            return super().mark_trial_as_complete(trial, value)

    def mark_trial_as_pruned(self, trial, value=None):

        with self.optuna_usage_sem:
            return super().mark_trial_as_pruned(trial, value)


    def mark_trial_as_failed(self, trial, value=None):

        with self.optuna_usage_sem:
            return super().mark_trial_as_failed(trial, value)

    def sample_trial(self):
        with self.optuna_usage_sem:
            return super().sample_trial()


    # RUNNING A TRIAL -----------------------------------------------------------------------
    


    def after_trial(self, study : optuna.Study, trial : optuna.trial.FrozenTrial):
        
        '''
        Called when a trial is over
        It is passed to optuna in the callbacks when the objective is defined
        '''        

        super().after_trial(study, trial)
                        
        component_group = self._get_loader_component_group(trial)
        component_group.unload_all_components_with_retries()

    
    #def _initialize_database(self):
    #    
    #    self.database_path = os.path.join(self.get_artifact_directory(), OPTUNA_STUDY_PATH)  # Path to the SQLite database file
    #    
    #    self.lg.writeLine(f"Trying to initialize database in path: {self.database_path}")
#
    #    self.storage = JournalStorage(JournalFileBackend(file_path=self.database_path))

  

    def _call_objective(self):
        self.run_trials(self.n_trials)
        
    def get_last_reported_step(self, trial : optuna.Trial, component_index=None):

        with self.results_logger_sem:
            return super().get_last_reported_step(trial, component_index)
    

    def check_if_should_prune_trial(self, trial, step):

        with self.optuna_usage_sem:
            return super().check_if_should_prune_trial(trial, step)