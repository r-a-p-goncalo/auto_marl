
# DEFAULT COMPONENTS -------------------------------------

from automarlcore.advanced_input_management import ComponentParameterSignature

from automarl.components.rl.policy.policy import Policy

from automarl.components.basic_components.state_management import StatefulComponent

from automarl.components.loggers.logger_component import ComponentWithLogging

# ACTUAL AGENT COMPONENT ---------------------------

from automarl.component import ParameterSignature, requires_input_process
from automarl.components.fundamentals.translator.translator import Translator

from automarl.utils.shapes_util import clone_shape, torch_zeros_for_space


def no_process_state_for_agent(state):
    return state


class AgentSchema(ComponentWithLogging, StatefulComponent):

    '''
    Represents an Agent in an RL environment.
    It will also contain its policy, model and any other necessary data.
    '''


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                        "name" : ParameterSignature(),
                       "device" : ParameterSignature(get_from_parent=True, ignore_at_serialization=True),
                                   
                       "state_shape" : ParameterSignature(default_value='', description='The shape received by the model, only used when the model was not passed already initialized', ignore_at_serialization=True),
                       "action_shape" : ParameterSignature(default_value='', description='Shape of the output of the model, only used when the model was not passed already', ignore_at_serialization=True),
                                              
                       "policy" : ComponentParameterSignature(
                            priority=100, description="The policy to use for the agent, if not defined it will be created using the policy_class and policy_input"
                       ),
                    
                    "state_translator" : ComponentParameterSignature(mandatory=False)

                       
                    }
        
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()

        self.lg.writeLine(f"Processing agent input with values {self.values}\n")
        
        self.device = self.get_input_value("device")
    
        self.state_shape = self.get_input_value("state_shape") #shape of the state as the environment sees it
                
        self.processed_state_shape = clone_shape(self.state_shape) #shape of the state after # CORRECT THIS TO CLONE THE STATE AND NOT COPY REFERENCE
        
        self.action_output_shape = self.get_input_value("action_shape") #shape of the action

        self.initialize_state_translator()
        self.initialize_state_memory()
        self.initialize_policy()
        self.initialize_necessary_cache()

        self.lg.writeLine(f"Finished processing basic agent input with values {self.values}\n")

    
    def initialize_state_translator(self):

        self.state_translator : Translator = self.get_input_value("state_translator")

        if self.state_translator is not None:

            self.state_translator.pass_input({"original_shape" : self.state_shape["observation"]})

            self.state_translator.process_input_if_not_processed()

            self.lg.writeLine(f"Agent has state translator: {self.state_translator}")
            self.process_env_state = self.state_translator.translate_state
            
            self.processed_state_shape["observation"] = self.state_translator.get_shape(self.state_shape["observation"])

            self.lg.writeLine(f"Model input shape was translated from {self.state_shape} to {self.processed_state_shape}")


        else:
            self.lg.writeLine(f"Agent has no state translator")
            self.process_env_state = no_process_state_for_agent
        

    def initialize_state_memory(self):
        
        self.lg.writeLine(f"Initializing state memory, the agent will remember the last state for convenience in computations, but it will not be used by itself")
        self.state_memory = {} 
        

    
    def initialize_necessary_cache(self):
        
        self.lg.writeLine(f"Agent is allocating memory for its computations, using two tensors with shape {self.processed_state_shape}")
        
        self.state_memory["observation"] = self.allocate_tensor_for_state() # makes a list of tensors for the state_memory, using them to store memory of the states
        
    
    
    def allocate_tensor_for_state(self):
        
        return torch_zeros_for_space(self.processed_state_shape["observation"], device=self.device)


        
    def initialize_policy(self):
        
        self.lg.writeLine("Initializing policy...")    
        
        self.policy : Policy = self.get_input_value("policy", look_in_value_with_key="policy", look_in_attribute_with_name="policy")

        self.lg.writeLine(f"Will initialize policy with state input shape: {self.processed_state_shape} and action output shape: {self.action_output_shape}")
         
        self.policy.pass_input({"state_shape" : self.processed_state_shape,
                               "action_shape" : self.action_output_shape,})
        
    
    
    
    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    def get_policy(self):
        '''
        Returns the policy of the agent
        '''
        return self.policy
    
    @requires_input_process
    def policy_predict(self, state):
        
        '''makes a prediction based on the new state for a new action, using the current memory if need be'''
        
        state = {**state}
        state["observation"] = self.process_env_state(state["observation"])
        
        to_return = self.policy.predict(state)

        return to_return
    
    @requires_input_process
    def policy_predict_with_memory(self):
        
        '''makes a prediction for a new action, using the current memory'''
        
        return self.policy.predict(self.state_memory)
    
    @requires_input_process
    def call_policy_method(self, policy_method, state):
        
        '''calls the method of the policy with this Agent's state management strategy'''
        
        state = {**state}
        state["observation"] = self.process_env_state(state["observation"])
        return policy_method(state)
    

    @requires_input_process
    def call_policy_method_with_memory(self, policy_method):
        
        '''calls the method of the policy with this Agent's state management strategy'''
        
        return policy_method(self.state_memory)
                    
    
    @requires_input_process
    def policy_random_predict(self, state):
        '''
        Uses the policies's random prediction strategy to return an action
        '''
        to_return =  self.policy.random_prediction(state)

        return to_return
          
    
    # STATE MEMORY --------------------------------------------------------------------
            
    
    @requires_input_process
    def reset_agent_in_environment(self, initial_state): # resets anything the agent has saved regarding the environment
        '''
        Resets an agent in the environment, mainly making it remember a state in memory
        '''

        self.update_state_memory(initial_state)
    

    @requires_input_process    
    def update_state_memory(self, new_state): #update memory shared
        '''
        Makes the agent remember a new state
        '''

        new_state = {**new_state}
        new_state_obs = new_state.pop("observation")

        self.state_memory["observation"].copy_(self.process_env_state(new_state_obs))

        self.state_memory.update(new_state)
        

    @requires_input_process
    def get_current_state_in_memory(self):
        
        '''
        Gets current state in memory, note this returns the actual tensor of the state memory
        It should be cloned if needed
        '''
                
        return self.state_memory
             

    @requires_input_process 
    def get_internal_agent_state_with_new(self, new_state):
        '''Gets the agent's internal state taking into account a new state received by the environment'''
        return new_state
        

