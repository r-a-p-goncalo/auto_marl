from automarl.component import Component, ParameterSignature, requires_input_process


from automarlcore.advanced_input_management import ComponentParameterSignature, ComponentListParameterSignature
from automarl.components.fundamentals.acessories import AcessoryComponent
from automarl.components.ml.memory.memory_utils import interpret_values, interpret_values
from automarl.components.rl.agent.agent_components import AgentSchema
import torch


class NoAgentLearner(Component):
    '''
    A learner that does not directly target an agent
    '''

    parameters_signature = {
        "optimizations_per_learn" : ParameterSignature(default_value=1,custom_dict={
                                    "hyperparameter_suggestion" : ("int", {"low" : 1, "high" : 32})
                                }),

        "learning_acessories" : ComponentListParameterSignature(mandatory=False),

    }

        
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
        
        self.optimizations_per_learn : int = self.get_input_value("optimizations_per_learn")

        self.learning_acessories : list[AcessoryComponent] = self.get_input_value("learning_acessories")

        if self.learning_acessories is None:
            self.learning_acessories = []

        else:
            for acessory in self.learning_acessories:
                acessory.pass_input({"affected_component" : self})

    def _learn(self, trajectory):
        pass


    @requires_input_process
    def learn(self, trajectory) -> None:
        
        '''
            Learns the policy using the current policy and the trajectory it is supposed to learn
            
            Remember that On-Policy algorithms expect the trajectory to have been generated
            
            Args:
                trajectory: batch of transitions [ (all states), (all actions), (all next states), (all rewards) ]
                
        '''

        for acessory in self.learning_acessories:
            acessory.pre_fun()
        
        for _ in range(self.optimizations_per_learn):
            values = self._learn(trajectory)

        for acessory in self.learning_acessories:
            acessory.pos_fun(values)


class LearnerSchema(NoAgentLearner):

    '''
    A learner that learns behavior of an Agent
    '''
        
    parameters_signature = {
        "agent" : ParameterSignature(),

        "agent_trainer" : ComponentParameterSignature(mandatory=False),


    }
        
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
        
        self.agent : AgentSchema = self.get_input_value("agent")

        self.agent_trainer = self.get_input_value("agent_trainer")



    @requires_input_process
    def interpret_trajectory(self, trajectory):

        interpreted_trajectory = {**trajectory}
        
        interpreted_trajectory["observation"] = interpret_values(trajectory["observation"], self.device).detach()

        interpreted_trajectory["action"] = interpret_values(trajectory["action"], self.device).detach()

        interpreted_trajectory["next_observation"] = interpret_values(trajectory["next_observation"], self.device).detach()
            
        interpreted_trajectory["reward"] = interpret_values(trajectory["reward"], self.device).detach()

        interpreted_trajectory["done"]  = interpret_values(trajectory["done"], self.device).detach()
            
        return interpreted_trajectory
    