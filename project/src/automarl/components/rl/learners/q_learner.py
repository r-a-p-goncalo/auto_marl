
from automarl.component import ParameterSignature, requires_input_process

from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.ml.models.model_components import ModelComponent
from automarl.components.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automarl.components.rl.learners.learner_component import LearnerSchema

from automarl.components.rl.policy.qpolicy import QPolicy
import torch

from automarl.components.rl.policy.policy import Policy


class QLearnerSchema(LearnerSchema, ComponentWithLogging):
    
    '''
    This represents a Deep Q Learner
    It has decouples the prediction of the q values by having not only having the policy network but also a target network
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {

                "discount_factor" : ParameterSignature(get_from_parent=True)
                        }    
    

    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
                
        self.discount_factor = self.get_input_value("discount_factor")
        
        
        
    def initialize_optimizer(self):
        
        self.optimizer : OptimizerSchema = self.get_input_value("optimizer")        
        self.optimizer.pass_input({"model" : self.model})

    
    # EXPOSED METHODS --------------------------------------------------------------------------

    def _apply_model_prediction_given_state_action_pairs(self, interpreted_trajectory):

        '''Returns the values predicted by the current model and the values for the specific actions that were passed'''
        pass
    

    def _apply_value_prediction_to_next_state(self, interpreted_trajectory):

        '''
        Returns the predicted values for the next state
        
        They are given by appying the Q function to them and then chosing the next 

        '''
        pass
    

    def _calculate_chosen_actions_correct_q_values(self, next_state_v_values, discount_factor, reward_batch):
        pass
    
    def _optimize_with_predicted_model_values_and_correct_values(self, predicted_values, correct_values):
        pass
    
    def _learn(self, trajectory) -> None:
        
        super()._learn(trajectory)

        interpreted_trajectory = self.interpret_trajectory(trajectory)

        observation_batch = interpreted_trajectory["observation"]
        action_batch = interpreted_trajectory["action"]
        next_observation_batch = interpreted_trajectory["next_observation"]
        reward_batch = interpreted_trajectory["reward"]
        done_batch = interpreted_trajectory["done"]
                    
        predicted_actions_values, state_action_values = self._apply_model_prediction_given_state_action_pairs(interpreted_trajectory) 

        next_state_q_values, next_state_v_values = self._apply_value_prediction_to_next_state(interpreted_trajectory)

        correct_q_values_for_chosen_action = self._calculate_chosen_actions_correct_q_values(next_state_v_values, self.discount_factor, reward_batch)
                        
        self._optimize_with_predicted_model_values_and_correct_values(state_action_values, correct_q_values_for_chosen_action)  
        

        


class DeepQLearnerSchema(QLearnerSchema):
    
    '''
    This represents a Deep Q Learner
    It has decouples the prediction of the q values by having not only having the policy network but also a target network
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "target_update_rate" : ParameterSignature(
                            default_value=0.05,
                            custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.05, "high": 0.9 }]}
                            ),

                        "target_update_learn_interval" : ParameterSignature(default_value=1, description="How many optimization times before we update the target model",
                                                                        custom_dict={"hyperparameter_suggestion" : [ "int", {"low": 1, "high": 10 }]}),
                        
                        "device" : ParameterSignature(ignore_at_serialization=True, get_from_parent=True),
                        
                        "optimizer" : ComponentParameterSignature(
                            default_component_definition=(
                                AdamOptimizer,
                                {}
                            )
                            ),
                        
                        "target_policy" : ParameterSignature(mandatory=False, possible_types=[QPolicy], description="The target network if it already exists, a clone of the network of the policy")

                        }    
    
    exposed_values = {
        "target_policy" : 0
    }
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
                
        self.device = self.get_input_value("device")
        
        self.TAU = self.get_input_value("target_update_rate") #the update rate of the target network
        
        self.target_update_learn_interval = self.get_input_value("target_update_learn_interval")
        
        self.policy : Policy = self.agent.get_policy()

        self.custom_data_beyond_obs = [key for key in self.policy.input_state_shape.keys() if key != "observation"]
        
        self.model = self.policy.get_model()
        
        self.number_optimizations_done = 0
        
        self.initialize_target_network()
        self.initialize_optimizer()
        


    def initialize_target_network(self):
        
        if self.values["target_policy"] != 0:
                        
            self.lg.writeLine("Target network found already in exposed values, using that...")
            self.target_policy : QPolicy = self.values["target_policy"]
            self.target_net = self.target_policy.get_model()

        elif "target_policy" in self.input.keys():
            
            self.lg.writeLine("There was already a target network defined in the input")
            self.target_policy : QPolicy  = self.get_input_value("target_policy")
            self.values["target_policy"] = self.target_policy
            self.target_net = self.target_policy.get_model()
            
        else:
            self.lg.writeLine("No target network previously defined, creating a new one...")

            self.target_policy = self.policy.clone(
                save_in_parent=False,
                input_for_clone=
                    {"name" : "target_policy", 
                    "base_directory" : self,
                    "create_new_directory" : False}, 
                is_deep_clone=True)
            
            self.define_component_as_child(self.target_policy)
            
            self.target_net : ModelComponent = self.model.clone(
                save_in_parent=False,
                input_for_clone=
                    {"name" : "target_network", 
                    "base_directory" : self.target_policy,
                    "create_new_directory" : False}, 
                is_deep_clone=True)
            
            self.target_policy.define_component_as_child(self.target_net)
            self.target_policy.values["model"] = 0
            self.target_policy.input.pop("model", None)

            self.target_policy.pass_input({"model" : self.target_net})
            
            self.values["target_policy"] = self.target_policy

        
        
    def initialize_optimizer(self):
        
        self.optimizer : OptimizerSchema = self.get_input_value("optimizer")        
        self.optimizer.pass_input({"model" : self.model})

    def _state_batch_from_interpreted_trajectory(self, interpreted_trajectory):

        state_batch = {
            "observation" : interpreted_trajectory["observation"]
        }

        for key in self.custom_data_beyond_obs:
            state_batch[key] = interpreted_trajectory[key]

        return state_batch
    
    def _next_state_batch_from_interpreted_trajectory(self, interpreted_trajectory):

        next_state_batch = {
            "observation" : interpreted_trajectory["next_observation"]
        }

        for key in self.custom_data_beyond_obs:
            next_state_batch[key] = interpreted_trajectory[f"next_{key}"]

        return next_state_batch
    
    # EXPOSED METHODS --------------------------------------------------------------------------

    def _apply_model_prediction_given_state_action_pairs(self, interpreted_trajectory):

        '''Returns the values predicted by the current model and the values for the specific actions that were passed'''

        action_batch = interpreted_trajectory["action"]

        state_batch = self._state_batch_from_interpreted_trajectory(interpreted_trajectory)

        predicted_actions_values = self.policy.predict_model_output(state_batch)
        predicted_values_for_actions = predicted_actions_values.gather(1, action_batch) 

        return predicted_actions_values, predicted_values_for_actions
    

    def _apply_value_prediction_to_next_state(self, interpreted_trajectory):

        '''
        Returns the predicted values for the next state
        
        They are given by appying the Q function to them and then chosing the next 

        '''

        done_batch = interpreted_trajectory["done"]

        next_state_batch = self._next_state_batch_from_interpreted_trajectory(interpreted_trajectory)

        with torch.no_grad():
            next_state_q_values = self.target_policy.predict_model_output(next_state_batch) # it returns the maximum q-action values of the next action
            print(f"Next state q values shape: {next_state_q_values.shape}")
            # these are the max values of the q values of the next action, representing the action that would be chosen
            next_state_v_values = next_state_q_values.max(dim=1, keepdim=True).values
            next_state_v_values = next_state_v_values * (1 - done_batch) # they are ignored when there is no next state

        return next_state_q_values, next_state_v_values
    

    def _calculate_chosen_actions_correct_q_values(self, next_state_v_values, discount_factor, reward_batch):

        correct_q_values_for_chosen_action = next_state_v_values * discount_factor + reward_batch

        return correct_q_values_for_chosen_action.detach()
    
    def _optimize_with_predicted_model_values_and_correct_values(self, predicted_values, correct_values):
    
        #Optimizes the model given the optimizer defined
        self.optimizer.optimize_model(predicted_values, correct_values)        
        
        self.number_optimizations_done += 1
    
    def _learn(self, trajectory) -> None:
        
        super()._learn(trajectory)

        if self.number_optimizations_done % self.target_update_learn_interval == 0:
            self.update_target_model()

        
        
    @requires_input_process            
    def update_target_model(self):
        
        self.target_net.update_model_with_target(self.model, self.TAU)



class DoubleDeepQLearnerSchema(DeepQLearnerSchema):


    def _apply_value_prediction_to_next_state(self, interpreted_trajectory):

        next_state_batch = self._next_state_batch_from_interpreted_trajectory(interpreted_trajectory)

        done_batch = interpreted_trajectory["done"]

        with torch.no_grad():

            q_values_that_would_be_given_next = self.policy.predict_model_output(next_state_batch)
            actions_that_would_be_chosen_next = q_values_that_would_be_given_next.argmax(1, keepdim=True)

            next_state_q_values = self.target_policy.predict_model_output(next_state_batch) # it returns the maximum q-action values of the next action
            
            next_state_v_values = next_state_q_values.gather(1, actions_that_would_be_chosen_next).squeeze(1)
            next_state_v_values = next_state_v_values * (1 - done_batch)

        return next_state_q_values, next_state_v_values