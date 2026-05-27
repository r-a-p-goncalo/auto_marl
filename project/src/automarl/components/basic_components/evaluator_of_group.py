

from automarl.components.basic_components.component_group import RunnableComponentGroup
from automarl.components.basic_components.evaluator_component import EvaluatorComponent
from automarl.component import Component, requires_input_process
from automarlcore.advanced_input_management import ComponentParameterSignature
from automarlcore.input_management import ParameterSignature
from automarl.components.loggers.result_logger import ResultLogger
import pandas

from automarl.components.loggers.global_logger import globalWriteLine



class EvaluatorComponentOfGroup(EvaluatorComponent):
    
    '''
    A component that evaluates a Component, being able to give it a single numeric score
    It may take into account not only its results, but also other things like its complexity
    
    '''
    
    parameters_signature = {
        "std_deviation_factor" : ParameterSignature(default_value=4, description="The factor to be used to calculate the standard deviation"),
        "base_evaluator" : ComponentParameterSignature(mandatory=False)
    }

    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.std_deviation_factor = self.get_input_value("std_deviation_factor")

        self.base_evaluator = self.get_input_value("base_evaluator", look_in_attribute_with_name="base_evaluator")
        
        if  self.base_evaluator ==None:
            globalWriteLine(f"WARNING: no base evaluator defined for evaluator component of group, this needs all the components in the group to be defined with the same evaluator or it will not work as intended")


    # EVALUATION -------------------------------------------------------------------------------
    
    @requires_input_process
    def _evaluate(self, component_to_evaluate : RunnableComponentGroup) -> dict:
        '''
        Evaluates a component group
        '''
        if not isinstance(component_to_evaluate, RunnableComponentGroup):
            raise TypeError("The component to evaluate must be a RunnableComponentGroup")
        
        if not self.base_evaluator == None:
            raise NotImplementedError()
        
        results_lg_of_group : ResultLogger = component_to_evaluate.get_aggregate_results_lg()
        
        dataframe_of_group : pandas.DataFrame = results_lg_of_group.get_dataframe()
        
        max_times_ran = dataframe_of_group["times_ran"].max()

        # Filter rows where times_ran == max_times_ran
        df_max_times = dataframe_of_group[dataframe_of_group["times_ran"] == max_times_ran]

        # For each component_index, keep the row with max_times_ran
        df_max_grouped = df_max_times.groupby("component_index").first().reset_index()

        mean_result = df_max_grouped["result"].mean()
        std_result = df_max_grouped["result"].std()
        
        result = mean_result - (std_result / self.std_deviation_factor)
        
        return {"result" : result}
        
