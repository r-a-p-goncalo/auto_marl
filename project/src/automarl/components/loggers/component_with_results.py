
from typing import Union
from automarl.component import Component, requires_input_process
from automarl.components.basic_components.artifact_management import ArtifactComponent
from automarlcore.input_management import ParameterSignature
from automarl.components.loggers.result_logger import ResultLogger, get_results_logger_from_file



DEFAULT_RESULTS_LOGGER_KEY = "results"


def on_log_pass(self : Component):
                
    if isinstance(self.input["results_logger_object"], dict):
        
        for key, logger_object in self.input["results_logger_object"].items():
            self.set_as_result_logger_object(logger_object, key)
    
    else:
        self.set_as_result_logger_object(self.input["results_logger_object"]) 
    
    
def generate_logger_for_component(self : ArtifactComponent):
    
    if not hasattr(self,"results_columns") or self.results_columns == None or self.results_columns == [] or self.results_columns == {}:
        return None
    
    else:
        
        for results_logger_key in self.results_loggers_names:
        
            self.set_as_result_logger_object(
                    self.initialize_child_component(ResultLogger, input={
                    "create_new_directory" : False,
                    "base_directory" : self.get_artifact_directory(), 
                    "artifact_relative_directory" : "",
                    "results_columns" : self.results_columns[results_logger_key] if results_logger_key in self.results_columns.keys() else [],
                    "results_filename" : f"{results_logger_key}.csv"}
                ),
                    results_logger_key
            )
        

# TODO: In the future, components may extend this, but not the LoggingComponent
#TODO: Components should be able to easily have multiple ResultLoggers
class ComponentWithResults(ArtifactComponent):

    '''
    A component that generalizes the behaviour of a component that has a results logger object 
    
    A component that extends it should 
    
    '''
    
    
    parameters_signature = {
                                                                                
                       "results_logger_object" : ParameterSignature(ignore_at_serialization=True, priority=10, 
                                                        generator = generate_logger_for_component , 
                                                        on_pass=on_log_pass,
                                                        description="A dictionary of result loggers object or single one",
                                                        mandatory=False),


                        "results_logger_input" : ParameterSignature(ignore_at_serialization=True, default_value={}),
                                                    
                                                
                       }
    
    
    #every class that extends ComponentWithResults should define this, so that the ResultLogger can be initialized with the right columns
    #if not or if it is empty, it will be responsibility of the class to initialize the ResultLogger
    results_columns : Union[list[str], dict[str, list]] = [] 
    
    results_loggers_names : list[str] = [DEFAULT_RESULTS_LOGGER_KEY] 


    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.__results_loggers : dict[str, ResultLogger] = {} # all results loggers (each with a different dataframe)


        if isinstance(self.results_columns, list): # if the results_columns passed had no specified key
            self.results_columns = {DEFAULT_RESULTS_LOGGER_KEY : self.results_columns} 
        
    


    def _process_input_internal(self): #this is the best method to have initialization done right after
            
        super()._process_input_internal()
    
        self.results_logger_input = self.get_input_value("results_logger_input")
        self.pass_input_to_results_logger(self.results_logger_input)


                
    def set_as_result_logger_object(self, logger_object, key=DEFAULT_RESULTS_LOGGER_KEY):
        '''Sets a results logger object as one of the results loggers for this component, with the specified key'''
        self.__results_loggers[key] = logger_object

    
    def pass_input_to_results_logger(self, input_to_pass, key=DEFAULT_RESULTS_LOGGER_KEY):
        '''Passes input for one of the results logger'''
        self.__results_loggers[key].pass_input(input_to_pass)


    def remove_result_logger_object(self, key):
        '''Removes one of the results loggers for this component, with the specified key'''
        self.__results_loggers.pop(key)
        self.results_columns.pop(key)
    
    
    def add_to_columns_of_results_logger(self, new_columns : list, key=DEFAULT_RESULTS_LOGGER_KEY):
        
        '''Add columns to a results logger of this object. Note that this should only be used on '''

        results_logger = self.__results_loggers[key]
        
        if results_logger.input_was_processed():
            raise Exception("Trying to add columns to results logger that was already used")

        else:
            results_logger.pass_input({"results_columns": [*results_logger.input["results_columns"], *new_columns]})
        

    @requires_input_process
    def get_results_columns(self, key=DEFAULT_RESULTS_LOGGER_KEY):
        return self.__results_loggers[key].get_results_columns()
    
    
    
    @requires_input_process
    def log_results(self, results, key=DEFAULT_RESULTS_LOGGER_KEY):
        '''
            Used internally to log the results of a component with results
        '''
                
        self.__results_loggers[key].log_results(results)
        
        
    
    @requires_input_process 
    def calculate_results(self) -> dict:
        '''
            Used internally to calculate the results of a component with results
        '''
        raise NotImplementedError("This method should be implemented in the child class")
    
    

    def calculate_and_log_results(self, key=DEFAULT_RESULTS_LOGGER_KEY):
        '''
            Used internally to calculate the results of a component with results and then log them
        '''
        
        self.log_results(self.calculate_results(), key)
    
    @requires_input_process
    def get_last_results(self, key=DEFAULT_RESULTS_LOGGER_KEY):
        
        return self.__results_loggers[key].get_last_results()
    
    @requires_input_process
    def get_results_logger(self, key=DEFAULT_RESULTS_LOGGER_KEY) -> ResultLogger:
        
        return self.__results_loggers[key]
    
    def save_dataframe(self, key=None):

        if key == None:
            for results_logger in self.__results_loggers.values():
                results_logger.save_dataframe()

        else:
            self.__results_loggers[key].save_dataframe()
    

    def get_decoupled_results_logger(self, dataframe_file='results.csv') -> ResultLogger:
        
        '''
        Returns the results logger
        This does not need the component to be initialized, as it is capable of getting them from file if needed
        '''
        
        return get_results_logger_from_file(self.get_artifact_directory(), dataframe_file)


def save_all_dataframes_of_component_and_children(component : Component):

    if isinstance(component, ComponentWithResults) or isinstance(component, ResultLogger):
        component.save_dataframe()

    for child_component in component.child_components:
        save_all_dataframes_of_component_and_children(child_component)