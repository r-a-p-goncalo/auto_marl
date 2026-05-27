from automarl.component import requires_input_process
from automarl.core.advanced_input_management import ComponentParameterSignature

from automarl.components.rl.exploration.exploration_strategy import ExplorationStrategySchema
from automarl.components.rl.trainers.agent_trainer_component import AgentTrainer

from automarl.components.rl.learners.q_learner import DeepQLearnerSchema
import torch

from automarl.utils.shapes_util import reduce_space_dimension

class AgentTrainerDQN(AgentTrainer):
    
    '''Describes a trainer specific for an agent, using a learner algorithm, memory and more'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {

                       
                       "learner" : ComponentParameterSignature(
                            default_component_definition=(DeepQLearnerSchema, {})
                        ),

        
                        "exploration_strategy" : ComponentParameterSignature(mandatory=False
                                                        ),
                       
                       }        
        

    # INITIALIZATION ---------------------------------------------
    
    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.initialize_exploration_strategy()
        

    def initialize_exploration_strategy(self):
                
        if "exploration_strategy" in self.input.keys():
            
            self.exploration_strategy : ExplorationStrategySchema = self.get_input_value("exploration_strategy") 
            self.exploration_strategy.pass_input(input= {"training_context" : self}) #the exploration strategy has access to the same training context
        
            self.lg.writeLine(f"Using exploration strategy {self.exploration_strategy.name}")
            
        else:
            self.lg.writeLine(f"No exploration strategy defined")
            
            self.exploration_strategy = None

        

    def initialize_memory(self):
        
        super().initialize_memory()

        state_shape = {**self.agent.processed_state_shape}

        observation_shape = state_shape.pop("observation")
        
        self.memory_fields_shapes = [   *self.memory_fields_shapes, 
                                        ("observation", observation_shape), 
                                        ("action", self.agent.get_policy().get_policy_output_shape(), torch.int64),
                                        ("next_observation", observation_shape),
                                        ("reward", 1),
                                        ("done", 1),
                                        *[(state_shape_key, state_shape_value) for state_shape_key, state_shape_value in state_shape.items()],
                                        *[(f"next_{state_shape_key}", state_shape_value) for state_shape_key, state_shape_value in state_shape.items()]
                                    ]
            
        self.memory.pass_input({
                                    "transition_data" : self.memory_fields_shapes
                                })
        

    @requires_input_process
    def end_training(self):        
        super().end_training()
        
        if self.exploration_strategy is not None:
            self.lg.writeLine(f"Exploration strategy values: \n{self.exploration_strategy.values}\n")
        

    def _observe_transition_to(self, prev_state, new_state, action, reward, done, truncated):
        
        '''Makes agent observe and remember a transiction from its (current) a state to another'''

        prev_state_in_agent = {**prev_state}
        prev_state_in_agent.pop("observation")

        for k, v in prev_state_in_agent.items():
            if not isinstance(v, torch.Tensor):
                prev_state_in_agent[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
        
        next_state_in_agent = {**new_state}
        next_state_in_agent.pop("observation")

        processed_next_state_in_agent = {}

        for k, v in next_state_in_agent.items():
            if not isinstance(v, torch.Tensor):
                processed_next_state_in_agent[f"next_{k}"] = torch.tensor(v, dtype=torch.float32, device=self.device)

                
        self.push_to_memory({"observation" : prev_state["observation"], 
                          "action" : action, 
                          "next_observation" : new_state["observation"], 
                          "reward" : reward, 
                          "done" : done,
                          **prev_state_in_agent,
                          **processed_next_state_in_agent})
        

        

    def select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''


        if self.exploration_strategy is not None:
  
            return self.exploration_strategy.select_action(self.agent, state)  

        else:
            return super().select_action(state)
        

    def select_action_with_memory(self):

        if self.exploration_strategy is not None:
  
            return self.exploration_strategy.select_action_with_memory(self.agent)  

        else:
            return self.agent.policy_predict_with_memory()
        
        