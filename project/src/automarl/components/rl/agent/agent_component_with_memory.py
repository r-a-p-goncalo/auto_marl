

from automarl.component import requires_input_process
from automarlcore.input_management import ParameterSignature
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.utils.shapes_util import torch_shape_from_space
import torch


class AgentSchemaWithStateMemory(AgentSchema):
    
    '''
    An agent which state has memory of previous states, such as a robot remembering its previous positions
    '''


    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = { 
                       
                       "state_memory_size" : ParameterSignature(default_value=2, description="This makes the agent remember previous states of the environment and concatenates them"),

                    }

        
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
            

    def initialize_state_memory(self): #Note this overrides the super method
        
        super().initialize_state_memory()

        self.model_input_shape_no_memory = self.processed_state_shape["observation"]

        self.state_memory_size = self.get_input_value("state_memory_size")
                
        if self.state_memory_size <= 1:
            raise Exception("State memory size must be greater than 1 to use this agent schema")
        
        self.processed_state_shape["observation"] = (self.state_memory_size, *self.model_input_shape_no_memory)

        self.lg.writeLine(f"State memory is {self.state_memory_size}, this means the agent will remember the last {self.state_memory_size - 1} states besides the new one it is experiencing")
     
        self.lg.writeLine(f"Input shape for observation is then transformed <{self.model_input_shape_no_memory}> -> <{self.processed_state_shape['observation']}>")

        if torch.cuda.is_available():   
            self.lg.writeLine(f"Initialized state memory, Cuda memory allocated: {torch.cuda.memory_allocated()}, Cuda memory reserved: {torch.cuda.memory_reserved()}, Cuda memory available: {torch.cuda.mem_get_info()}")
        
        else:
            self.lg.writeLine(f"Initialized state memory, Cuda not available, using CPU memory")
        
        
    def initialize_necessary_cache(self):
        
        super().initialize_necessary_cache()

        self.temp_cache_state_memory = self.allocate_tensor_for_state() # a reserved memory space to store 


    
    # EXPOSED TRAINING METHODS -----------------------------------------------------------------------------------
    
    @requires_input_process
    def policy_predict(self, state):
        
        '''
        Makes a prediction based on the new state for a new action, using the current memory
        Note that this does not observe the transition
        '''
                
        state = {**state}
        new_obs = self.process_env_state(state["observation"])
        state["observation"] = self.get_internal_agent_state_with_new(new_obs)
        return self.policy.predict(state)
    
    @requires_input_process
    def call_policy_method(self, policy_method, state):
        
        '''calls the method of the policy with this Agent's state management strategy'''
        
        state = {**state}
        new_obs = self.process_env_state(state["observation"])
        state["observation"] = self.get_internal_agent_state_with_new(new_obs)
        return policy_method(state)
        
    
    # STATE MEMORY --------------------------------------------------------------------
            
    
    @requires_input_process
    def reset_agent_in_environment(self, initial_state : torch.Tensor): #setup memory shared accross agents

        initial_state = {**initial_state}
        obs = self.process_env_state(initial_state.pop("observation"))

        self.state_memory["observation"] = (
            obs.unsqueeze(0)
               .expand(self.state_memory_size, *self.model_input_shape_no_memory)
               .clone()
        )

        self.state_memory.update(initial_state)
         
    @requires_input_process    
    def update_state_memory(self, new_state): #update memory of agent
        '''Updates memory of agent with new state'''
        
        new_state = {**new_state}
        new_obs = self.process_env_state(new_state.pop("observation"))

        self.state_memory["observation"].copy_(self.get_internal_agent_state_with_new(new_obs))
        self.state_memory.update(new_state)
        
       
    def get_internal_agent_state_with_new(self, new_state):
        '''
        Returns a new state memory tensor with the new_state added,
        shifting the previous states to the left.
        
        Note that this tensor is cloned and not used anymore by the agent, so it can be safely used
        '''
        with torch.no_grad():

            # shift previous memory left by one position
            self.temp_cache_state_memory[:-1].copy_(self.state_memory["observation"][1:]) #note that this strategy does not work well with autograd, as this can be changed after it was used to compute a tensor
    
            # insert new state into the last position
            self.temp_cache_state_memory[-1].copy_(new_state)
        
        return self.temp_cache_state_memory
    
         
        
        

