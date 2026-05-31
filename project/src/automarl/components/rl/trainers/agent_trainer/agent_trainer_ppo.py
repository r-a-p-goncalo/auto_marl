from automarl.components.ml.memory.memory_samplers.advantages_calc_sampler import PPOAdvantagesCalcSampler
from automarl.components.rl.policy.stochastic_policy import StochasticPolicy
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer

from automarl.components.rl.learners.ppo_learner import PPOLearner
from automarl.core.advanced_input_management import ComponentParameterSignature
import torch

class AgentTrainerPPO(AgentTrainer):
    
    '''Describes a trainer specific for an agent, using a learner algorithm, memory and more'''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {

                       "learner" : ComponentParameterSignature(
                            default_component_definition=(PPOLearner, {})
                        ),

                       "memory_transformer" : ComponentParameterSignature(default_component_definition=(PPOAdvantagesCalcSampler, {})),

                       }
    

    def _process_input_internal(self):
        
        super()._process_input_internal()
        

    # INITIALIZATION ---------------------------------------------
    

    def initialize_memory(self):
        
        super().initialize_memory()
                
        state_shape = {**self.agent.processed_state_shape}

        observation_shape = state_shape.pop("observation")

        self.memory_fields_shapes = [   *self.memory_fields_shapes, 
                                        ("observation", observation_shape), 
                                        ("action", self.agent_policy.get_policy_output_shape()),
                                        ("next_observation", observation_shape),
                                        ("reward", 1),
                                        ("done", 1),
                                        ("log_prob", 1), #log probability of chosing the stored action
                                        ("action_val", self.agent_policy.get_action_val_shape()),
                                        ("truncation", 1),
                                        *[(state_shape_key, state_shape_value) for state_shape_key, state_shape_value in state_shape.items()]
                                    ]
            
        self.memory.pass_input({
                                    "device" : self.device,
                                    "transition_data" : self.memory_fields_shapes
                                })
        
        self.memory.clear() # IN PPO the memory must be filled only with transitions the current policy did

        if self.memory_transformer is not None:
            self.memory_transformer.pass_input({
                "learner" : self.learner,
                "discount_factor" : self.discount_factor
                })
        else:
            self.lg.writeLine(f"Note that PPO assumes advantage estimation rollot but no memory transformer was passed")

    def initialize_agent(self):
        
        super().initialize_agent()

        if not isinstance(self.agent_policy, StochasticPolicy):
            raise Exception("PPO trainer needs a stochastic policy")
        
        self.agent_policy : StochasticPolicy = self.agent_policy



        
    
    # TRAINING_PROCESS ---------------------

    def _gen_to_push(self, prev_state, next_state, action, reward, done, truncated):

        to_push = super()._gen_to_push(prev_state, next_state, action, reward, done, truncated)

        prev_state_in_agent = {**prev_state}
        prev_state_in_agent.pop("observation")

        for k, v in prev_state_in_agent.items():
            if not isinstance(v, torch.Tensor):
                prev_state_in_agent[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
        
        action_val_to_store = self.last_action_val.squeeze(0) if torch.is_tensor(self.last_action_val) and self.last_action_val.dim() > 1 and self.last_action_val.shape[0] == 1 else self.last_action_val

        return {**to_push,
                            "observation" : prev_state["observation"], 
                              "action" : action, 
                              "next_observation" : next_state["observation"], 
                              "reward" : reward, 
                              "log_prob" : self.last_log_prob, 
                              "done" : done,
                              "action_val" : action_val_to_store,
                              "truncation" : truncated,
                              **prev_state_in_agent
                              }

               
        
    def select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''
                
        action_val, log_prob = self.agent.call_policy_method(self.agent_policy.predict_action_val_with_log, state) 
        
        self.last_action_val = action_val
        self.last_log_prob = log_prob
        
        return self.agent_policy.get_action_from_action_val(action_val)
    
    def select_action_with_memory(self):

        action_val, log_prob = self.agent.call_policy_method_with_memory(self.agent_policy.predict_action_val_with_log)
        
        self.last_log_prob = log_prob
        self.last_action_val = action_val

        return self.agent_policy.get_action_from_action_val(action_val)
        

    def optimizeAgent(self):

        '''Optimizes the agent for the specified number of times and then clears the memory'''
    
        super().optimizeAgent()
        
        self.memory.clear() # in PPO the policy must be filled only with transitions the current agent did       


class AgentTrainerPPOCriticAware(AgentTrainerPPO):

    def initialize_memory(self):
        super().initialize_memory()

        self.memory_fields_shapes = [   *self.memory_fields_shapes, 
                                        ("observation_old_critic_value", 1),
                                        ("next_obs_old_critic_value", 1), 
                                    ]
            
        self.memory.pass_input({
                                    "transition_data" : self.memory_fields_shapes
                                })
        



        
    def _gen_to_push(self, prev_state, next_state, action, reward, done, truncated, prev_critic_val, next_critic_val):
        
        transition_to_push = super()._gen_to_push(prev_state, next_state, action, reward, done, truncated)

        transition_to_push.update({
            "observation_old_critic_value" : prev_critic_val,
            "next_obs_old_critic_value" : next_critic_val
        })

        return transition_to_push
               