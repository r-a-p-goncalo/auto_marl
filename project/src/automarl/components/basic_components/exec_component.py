from automarl.components.loggers.component_with_results import save_all_dataframes_of_component_and_children
from automarl.components.basic_components.state_management import save_state
from automarl.utils.smart_enum import SmartEnum
from automarl.components.loggers.logger_component import flush_text_of_all_loggers_and_children
from ..component import ParameterSignature, Component, requires_input_process


from typing import final

class State(SmartEnum): #an enumerator to track state of executable component
        IDLE = 0
        RUNNING = 1
        OVER = 2
        ERROR = 4
        INTERRUPTED = 5

class StopExperiment(Exception):
    '''
    An exception which implies an earlier stopping of the experiment without it being an error
    Usually will mean an outside interruption
    '''
    pass

# EXECUTABLE COMPONENT --------------------------

class ExecComponent(Component):
    
    parameters_signature = {
        
            "times_to_run" : ParameterSignature(mandatory=False, description="The number of times to run the component"),
            "save_state_on_run_end" : ParameterSignature(default_value=True, ignore_at_serialization=True),
            "save_dataframes_on_run_end" : ParameterSignature(default_value=True, ignore_at_serialization=True),
            "save_values_in_execution" : ParameterSignature(default_value=True)
    
    }

    exposed_values = {"running_state" : State.IDLE, "times_ran" : 0, "values_in_execution" : None}    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.values["values_in_execution"] = []
    
    def _process_input_internal(self):
        super()._process_input_internal()

        self._times_to_run = self.get_input_value("times_to_run")
        
        self.__save_state_on_run_end = self.get_input_value("save_state_on_run_end")
        self.__save_dataframes_on_run_end = self.get_input_value("save_dataframes_on_run_end")

        self._received_signal_to_stop = False        

        self.save_values_in_execution = self.get_input_value("save_values_in_execution")

        self._current_execution = self.values["times_ran"] + 1

    # METHODS TO OVERRIDE --------------------------------

    def is_over(self):
        '''Returns true if the this component is over'''
        return State.equals_value(State.OVER, self.values["running_state"])

    def _is_over(self):
        '''
        Defines condition for execution to be considered over instead of idle
        The base condition is for the training to have been run more times than the ones defined
        '''

        isover = self._times_to_run != None and self.values["times_ran"] >= self._times_to_run

        return isover


    def _algorithm(self): #the algorithm of the component
        pass
    
    def _pre_algorithm(self): # a component may extend this for certain behaviours
        self.values["running_state"] = State.RUNNING
        self._received_signal_to_stop = False
        self._current_execution = self.values["times_ran"] + 1
        

    def _pos_algorithm(self): # a component may extend this for certain behaviours
        
        self.values["times_ran"] = self._current_execution

        self.values["running_state"] = State.IDLE
        
        self._test_to_see_if_algorithm_is_over()


    def _test_to_see_if_algorithm_is_over(self):

        '''Forces a test to see if the algorithm should be considered over'''

        if not State.equals_value(State.OVER, self.values["running_state"]):
            if self._is_over():
                self.values["running_state"] = State.OVER

       

    def _stop_earlier_signal_received(self):
        return self._received_signal_to_stop
    
    def stop_execution_earlier(self):
        '''
        Stops the execution of this component by just setting the flag
        The component will have to have its state checked for this to raise an exception
        '''
        self._received_signal_to_stop = True

    def _check_if_should_stop_execution_earlier(self):
        '''Checks if the component should stop its execution earlier, returning a boolean (meant to be extended)'''
        return False
    
    
    def check_if_should_stop_execution_earlier(self, raise_exception=True):

        '''Raises exception if the execution should stop earlier, meant to be used'''

        if self._stop_earlier_signal_received():
            if raise_exception:
                raise StopExperiment("Stopped signal received")
            else:
                return True
        
        elif self._check_if_should_stop_execution_earlier():
            self.stop_execution_earlier()
            if raise_exception:
                raise StopExperiment("Stopped signal received")
            else:
                return True
        
        else:
            return False
        
    def _on_earlier_interruption(self):
        '''Called when runnable component is interrupted earlier'''
        pass
    
    

    def __on_exception(self, exception):
        '''To be called when a non treated exception happens'''

        self._deal_with_exception(exception)
    
    def _deal_with_exception(self, exception):
        '''Called internally when an exception happens'''
        pass

    def _save_values_in_execution(self):

        values_to_save = {}

        for key, value in self.values.items():
            #if key not in ExecComponent.exposed_values.keys():
            if key != "values_in_execution":
                    values_to_save[key] = value

        if len(values_to_save) > 0:
            values_to_save["execution"] = self._current_execution

        self.values["values_in_execution"].append(values_to_save)


    def _on_exception_running(self, exception : Exception):
    
            if isinstance(exception, StopExperiment):
                self.values["running_state"] = State.INTERRUPTED
                self.stop_execution_earlier()
                self._on_earlier_interruption()


            else:
                self.values["running_state"] = State.ERROR
                self.__on_exception(exception)

            self.values["times_ran"] += 1

            raise exception
    

    def _on_algorithm_end(self):
            
            if self.__save_state_on_run_end:
                save_state(self)
            
            if self.__save_dataframes_on_run_end:
                save_all_dataframes_of_component_and_children(self)
                flush_text_of_all_loggers_and_children(self)
    
            self._save_values_in_execution()

    # RUNNABLE METHOD --------------------------------

    @requires_input_process
    @final
    def start_algorithm(self):

        '''
        This offers an interface for components to extend, when they want to control if the algorithm is running or not
        '''

        if self.values["running_state"] == State.RUNNING:
            raise Exception(f"{self.name}: Algorithm was started while being run, current state is: {self.values['running_state']}")

        try:
            self._pre_algorithm()
        
        except:
            self._on_exception_running()
            self._on_algorithm_end()

    @requires_input_process
    @final
    def end_algorithm(self):

        '''
        This offers an interface for components to extend, when they want to control if the algorithm is running or not
        '''

        if self.values["running_state"] != State.RUNNING:
            raise Exception(f"{self.name}: Algorithm was ended while not being run, current state is: {self.values['running_state']}")

        try:
            self._pos_algorithm()
        
        except:
            self._on_exception_running()

        finally:
            self._on_algorithm_end()

    
    
    
    @requires_input_process
    @final
    def run_all(self):
        
        '''Runs the algorithm the number of times specified in the input.'''
        
        if self._times_to_run == None:
            return self.run()
        
        else:
            for i in range(self._times_to_run - self.values["times_ran"]):
                self.run()
                
            return self.get_output()
    
    @requires_input_process
    @final
    def run(self): #the universal execution flow for all components
        
        '''Runs the specified algorithm once.'''
         
        try:
            self._pre_algorithm()
            self._algorithm()
            self._pos_algorithm()
            return self.get_output()
        
        except Exception as e:
            self._on_exception_running(e)

        finally:
            self._on_algorithm_end()
    
