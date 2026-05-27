
from automarl.component import Component, requires_input_process
from automarl.components.basic_components.evaluator_component import EvaluatorComponent
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.core.advanced_input_management import ComponentListParameterSignature
from automarl.components.rl.evaluators.rl_single_agent_evaluator import RlSingleAgentEvaluator
from automarl.components.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.environment.environment_components import EnvironmentComponent
from automarl.core.localizations import get_component_by_localization


class RLAgentIterEvaluator(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines
    '''
    
    parameters_signature = {
        "single_agent_evaluators" : ComponentListParameterSignature()
    }
    

    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.single_agent_evaluators : list[RlSingleAgentEvaluator] = self.get_input_value("single_agent_evaluators")
        
    @requires_input_process
    def get_metrics_strings(self) -> list[str]:

        environment : EnvironmentComponent = get_component_by_localization(self, ["relative", 
                                                                                [
                                                                                   ["__get_by_type__", {"type" : EnvironmentComponent}]
                                                                                ]
                                                                                ])

        agents_names : list[str] = environment.agents()

        to_return = []

        for single_agent_evaluator in self.single_agent_evaluators:
            metrics_of_eval = single_agent_evaluator.get_metrics_strings()
            
            for agent_name in agents_names:
            
                for metric in metrics_of_eval:

                    to_return.append(f"{single_agent_evaluator.name}_{agent_name}_{metric}") 
            
            to_return.append(f"{single_agent_evaluator.name}_result")

        to_return.append("result")

        return to_return
                
        

    
    # EVALUATION -------------------------------------------------------------------------------

    def make_evaluation_for_single_agent_evaluator(self, component_to_evaluate : RLPipelineComponent, agents_dict : dict[str, AgentSchema], single_agent_evaluator : RlSingleAgentEvaluator, directory_to_store_evaluation_prefix : str):
    
        evaluations = {}

        results_for_each_agent = []

        for agent_name in agents_dict.keys():
            single_agent_evaluator.pass_input({"directory_to_store_evaluation_prefix" : f"{directory_to_store_evaluation_prefix}\\{agent_name}"})
            single_agent_evaluator.pass_input({"agent_name" : agent_name})

            agent_evaluations = single_agent_evaluator.evaluate(component_to_evaluate)
            results_for_each_agent.append(agent_evaluations["result"])

            for k, v in agent_evaluations.items():
                evaluations[f"{single_agent_evaluator.name}_{agent_name}_{k}"] = v

        result = sum(results_for_each_agent) / len(results_for_each_agent)
        evaluations[f"{single_agent_evaluator.name}_result"] = result

        return evaluations, result
    

    def _evaluate(self, component_to_evaluate : RLPipelineComponent):

        agents_dict = component_to_evaluate.get_agents()
        
        evaluations = {}
        results = []


        for single_agent_evaluator in self.single_agent_evaluators:
            e, r = self.make_evaluation_for_single_agent_evaluator(
                component_to_evaluate=component_to_evaluate,
                agents_dict=agents_dict,
                single_agent_evaluator=single_agent_evaluator,
                directory_to_store_evaluation_prefix=f"evaluations\\{single_agent_evaluator.name}"
            )

            evaluations.update(e)
            results.append(r)

        evaluations["result"] = sum(results) / len(results)

        return evaluations