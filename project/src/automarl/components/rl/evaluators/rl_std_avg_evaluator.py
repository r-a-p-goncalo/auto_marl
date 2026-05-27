
from automarl.component import Component, requires_input_process
from automarl.components.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.loggers.result_logger import ResultLogger

from automarlcore.input_management import ParameterSignature
from automarl.components.loggers.global_logger import globalWriteLine

class LastValuesAvgStdEvaluator(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines, which evaluates the last results it has and uses them to compute a result, penalizing high variance and using the mean as the base value
    
    This is meant to be used not as a final evaluation of a component, but as an intermediary evaluator at training time.
    
    '''
    
    parameters_signature = {
        "n_results_to_use" : ParameterSignature(default_value=10),
        "std_deviation_factor" : ParameterSignature(default_value=4, description="The factor to be used to calculate the standard deviation"),
        "value_to_use" : ParameterSignature(default_value="episode_reward", description="The value to use for evaluation, note that a higher value is expected to be a better value")
    }
    

    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.n_results_to_use = self.get_input_value("n_results_to_use")
        self.std_deviation_factor = self.get_input_value("std_deviation_factor")
        self.value_to_use = self.get_input_value("value_to_use")


    # EVALUATION -------------------------------------------------------------------------------
    
    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), "result", "avg", "std"]
    
    @requires_input_process
    def _evaluate(self, component_to_evaluate : RLPipelineComponent | ResultLogger):
        
        if isinstance(component_to_evaluate, ResultLogger):
            results_logger = component_to_evaluate
        else:
            results_logger : ResultLogger = component_to_evaluate.get_results_logger() 
        
        return self._evaluate_from_results(results_logger)
    

    @requires_input_process
    def _evaluate_from_results(self, results_logger : ResultLogger): # TODO: check if more evaluators should have this
        
        results_logger.save_dataframe()
        
        if not self.value_to_use in results_logger.get_results_columns():
            raise Exception(f"The value to use '{self.value_to_use}' is not in the results columns of the results logger")
                
        n_results_to_use = self.n_results_to_use
        n_rows = results_logger.get_number_of_rows()
                
        if n_results_to_use > n_rows:
            globalWriteLine(f"WARNING: Results rows to use in evaluator {self.name} were {n_results_to_use}, which is higher than the total number of rows: {n_rows}")
            n_results_to_use = n_rows

        avg_result, std_result = results_logger.get_avg_and_std_n_last_results(n_results_to_use, self.value_to_use)

        result = avg_result - (std_result / self.std_deviation_factor)

        return {"result" : result, "avg" : avg_result, "std" : std_result}
