

from automarl.components.rl.evaluators.rl_single_agent_evaluator import RlSingleAgentEvaluator
from automarl.components.rl.evaluators.rl_evaluator_player import EvaluatorWithPlayer
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.policy.random_policy import RandomPolicy
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.core.input_management import ParameterSignature
from automarl.components.rl.policy.policy import Policy
from automarl.utils.class_util import get_class_from
from automarl.utils.json_utils.json_component_utils import gen_component_from
from automarl.component import requires_input_process




class AgentVsAgents(RlSingleAgentEvaluator, EvaluatorWithPlayer):

    '''
    Evaluates a single agent vs other agents
    '''

    def _process_input_internal(self):

        super()._process_input_internal()

        self.base_evaluator.pass_input({"value_to_use" : f"{self.agent_name}_reward"})
    
    @requires_input_process
    def get_metrics_strings(self) -> list[str]:

        return self.base_evaluator.get_metrics_strings()

    def _initialize_other_agent(self, rl_player : RLPlayer, agent_name : str, agent : AgentSchema, component_to_evaluate : RLPipelineComponent =None):
        pass


    def _initialize_player_to_run(self, agents : dict[str, AgentSchema], device, evaluations_directory, env, seed_for_player=None, component_to_evaluate : RLPipelineComponent = None):

        rl_player = super()._initialize_player_to_run(agents, device, evaluations_directory, env, seed_for_player, component_to_evaluate)

        agents = {**agents} # this is to be sure we're not changing a used dict

        for agent_name, agent in agents.items():

            if agent_name != self.agent_name:
                agents[agent_name] = self._initialize_other_agent(rl_player, agent_name, agent, component_to_evaluate)

        rl_player.pass_input({"agents" : agents})

        return rl_player
    


class AgentVsAgentsWithPolicy(AgentVsAgents):

    '''
    Evaluates the reward of the agent when the other agents are following a random policy
    '''

    parameters_signature = {
        "policy_type_for_others" : ParameterSignature(default_value=RandomPolicy),

    }
    

    

    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.policy_type_for_others : type[Policy] = get_class_from(self.get_input_value("policy_type_for_others"))


    def _initialize_other_agent(self, rl_player : RLPlayer, agent_name, agent : AgentSchema, component_to_evaluate : RLPipelineComponent =None):
                
        new_agent_input = {**agent.input}

        new_agent_input["policy"] = rl_player.initialize_child_component(self.policy_type_for_others, {})
                
        new_agent = rl_player.initialize_child_component(type(agent), new_agent_input)

        return new_agent




class AgentVsAgentsWithPreviousPolicy(AgentVsAgents):

    '''
    Evaluates the reward of the agent when the other agents are following a random policy
    '''

    parameters_signature = {
        "fall_back_policy_type_for_others" : ParameterSignature(default_value=RandomPolicy),
    }
    


    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.fall_back_policy_type_for_others : type[Policy] = get_class_from(self.get_input_value("fall_back_policy_type_for_others"))


    def _get_prev_component(self, component_to_evaluate : RLPipelineComponent):

        checkpoints = component_to_evaluate.values.get("checkpoints_and_evaluations")

        if len(checkpoints) == 0:
            self.prev_rl_pipeline : RLPipelineComponent = None

        else:  
            (last_checkpoint_path, _) = checkpoints[-1]

            self.prev_rl_pipeline : RLPipelineComponent = gen_component_from(last_checkpoint_path)



    def  _evaluate_agents(self, agents, device, evaluations_directory, env=None, component_to_evaluate=None):

        self._get_prev_component(component_to_evaluate)

        return super()._evaluate_agents(agents, device, evaluations_directory, env, component_to_evaluate)


    def _get_previous_agent(self, agent_name : str):

        if self.prev_rl_pipeline is None:
            return None 

        return self.prev_rl_pipeline.get_agents()[agent_name]


    def _initialize_other_agent(self, rl_player : RLPlayer, agent_name, agent : AgentSchema, component_to_evaluate : RLPipelineComponent =None):
                
        new_agent = self._get_previous_agent(agent_name)

        if new_agent is None:

            new_agent_input = {**agent.input}
            
            new_policy = rl_player.initialize_child_component(self.fall_back_policy_type_for_others, {})

            new_agent_input["policy"] = new_policy
                
            new_agent = rl_player.initialize_child_component(type(agent), new_agent_input)

        

        return new_agent
