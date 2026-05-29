

from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.core.input_management import ParameterSignature
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.memory.memory_components import MemoryComponent
from automarl.components.ml.memory.memory_samplers.memory_sampler import MemorySampler
from automarl.components.ml.memory.memory_samplers.advantages_calc_sampler import PPOAdvantagesCalcSampler
from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent


class RLTrainerOrquestrator(RLTrainerComponent):

    '''
    An RL Trainer which takes control of the training flow of the agents and can orquestrate the training using the global state

    It has its own memory, where it also stores transitions
    '''


    parameters_signature = {
        
                       "optimization_interval" : ParameterSignature(description="How many steps between optimizations", default_value=1,
                                                                custom_dict={"hyperparameter_suggestion" : [ "int", {"low": 100, "high": 500 }]}
                                                                ),


                       "learning_start_ep_delay" : ParameterSignature(default_value=-1),
                        "learning_start_step_delay" : ParameterSignature(default_value=-1),

                       "memory" : ComponentParameterSignature(
                            default_component_definition=(TorchMemoryComponent, {}),
                       ),

                       "memory_transformer" : ComponentParameterSignature(),

                       }
    

    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.optimization_interval = self.get_input_value("optimization_interval")

        if self.limit_steps >= 0:
            self.lg.writeLine(f"Will limit the learning of this agent to {self.limit_steps} steps in this training session")    
        
        self._initialize_delays()                  
                                
        self.initialize_memory()

    def _initialize_delays(self):
        
        self.learning_start_ep_delay = self.get_input_value("learning_start_ep_delay")
        self.learning_start_step_delay = self.get_input_value("learning_start_step_delay")

    
    def initialize_memory(self):
        
        self.memory : MemoryComponent = self.get_input_value("memory", look_in_attribute_with_name="memory")
        
        self.memory_fields_shapes = [] # tuples of (name_of_field, dimension)
            
        self.memory.pass_input({
                                    "device" : self.device,
                                })
        
        self.memory_transformer : MemorySampler = self.get_input_value("memory_transformer")

        if self.memory_transformer is not None:
            self.lg.writeLine(f"Will use memory transformer: {self.memory_transformer.name}")
    


    def setup_training_session(self):

        self.lg._writeLine(f"Starting to run training with number of episodes: {self.num_episodes} and total step limit: {self.limit_total_steps}")

        if self._fraction_training_to_do != None:

            if self._fraction_training_to_do <= 0 or self._fraction_training_to_do > 1:
                raise Exception(f"Fraction of training to do must be between 0 and 1, was {self._fraction_training_to_do}")

            self.lg._writeLine(f"Only doing a fraction of {self._fraction_training_to_do} of the training")

        self.lg._writeLine(f"Resetting the environment...")

        self.env.total_reset()

        self.lg.writeLine(f"Agent trainers will not be turned on, so internal algorithms to train will not be activated. Nevertheless, agent trainers have to collect synchronized memory")
        
        self.values["episodes_done_in_session"] = 0
        self.values["steps_done_in_session"] = 0


    def after_environment_step(self, reward):

        super().after_environment_step(reward)

        self._learn_if_needed()

    
    def _pre_agents_optimization(self):
        self._prepare_memory()


    def _optimize_agents(self):

        for agent_name, agent_trainer in self.agents_trainers.items():
            agent_trainer.optimizeAgent()


    def _pos_agents_optimization(self):
        self._pos_process_memory()

    def _prepare_memory(self):

        if self.memory_transformer is not None:
            self.memory_transformer.prepare(self.memory)

    def _pos_process_memory(self):
        if self.memory_transformer is not None:
            self.memory_transformer.let_go()

    def _learn_if_needed(self):

        '''
        Does a learning step 
        '''
        
        can_learn_by_ep_delay = self.learning_start_ep_delay < 1 or self.values["episodes_done"] >= self.learning_start_ep_delay
        can_learn_by_step_delay = self.learning_start_step_delay < 1 or self.values["total_steps"] >= self.learning_start_step_delay

        if can_learn_by_ep_delay and can_learn_by_step_delay:          
            
            if self.values["total_steps"] % self.optimization_interval == 0:
                self.optimize_agents()


    def optimize_agents(self):

        '''Does optimization of agents for the number of specified times''' 
            
        self._pre_agents_optimization()

        self._optimize_agents()

        self._pos_agents_optimization()
