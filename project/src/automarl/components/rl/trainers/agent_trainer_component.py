from automarl.components.basic_components.exec_component import ExecComponent
from automarl.component import ParameterSignature, requires_input_process
from automarlcore.advanced_input_management import ComponentParameterSignature, ComponentListParameterSignature
from automarl.components.fundamentals.acessories import AcessoryComponent
from automarl.components.loggers.component_with_results import ComponentWithResults
from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.ml.memory.memory_components import MemoryComponent
from automarl.components.ml.memory.memory_samplers.memory_sampler import MemorySampler
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent

from automarl.components.rl.learners.learner_component import LearnerSchema
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema
from automarl.components.rl.policy.policy import Policy
import torch

from automarl.components.basic_components.EventfulComponent import EventfulComponent, Event


class AgentTrainer(ComponentWithLogging, ComponentWithResults, EventfulComponent, ExecComponent):
    
    '''
    Describes a trainer specific for an agent, using a learner algorithm, memory and more

    It serves as a connection between an RLTrainer (which describes a general training algorithm for single or multi agent RL) and the agents

    It is resposible for knowing when the agent is learning or just being used, logging results, connecting the memory and the learning algorithm and so on

    Functionality such as using acessories is turned off when the agent is not learning
    '''
    
    TRAIN_LOG = 'train.txt'
    
    parameters_signature = {
        
                       "optimization_interval" : ParameterSignature(description="How many steps between optimizations", default_value=1,
                                                                custom_dict={"hyperparameter_suggestion" : [ "int", {"low": 100, "high": 500 }]}
                                                                ),
                       
                       "times_to_learn" : ParameterSignature(default_value=1, description="How many times to optimize at learning time",
                                                         custom_dict={"hyperparameter_suggestion" : [ "int", {"low": 1, "high": 256 }]}), 
                       
                       "learning_start_ep_delay" : ParameterSignature(default_value=-1),
                        "learning_start_step_delay" : ParameterSignature(default_value=-1),
                        
                       "device" : ParameterSignature(get_from_parent=True, ignore_at_serialization=True),
                            
                        "agent" : ComponentParameterSignature(),

                       "batch_size" : ParameterSignature(mandatory=False, custom_dict={"hyperparameter_suggestion" : [ "cat", {"choices": [8, 16, 32, 64, 128, 256]}]}),
                       
                       "learn_with_all_memory" : ParameterSignature(default_value=False, 
                                                                description="When true, each learning will consist of dividing the entire memory into batches with the specified size, and learning with each of them"
                                                                ),
                    
                       "discount_factor" : ParameterSignature(
                           default_value=0.95,
                           get_from_parent=True,
                           custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.7, "high": 0.99999 }]}
                           ),

                       "memory" : ComponentParameterSignature(
                            mandatory=False
                       ),

                       "memory_transformer" : ComponentParameterSignature(mandatory=False),
                       
                       "learner" : ComponentParameterSignature(
                            mandatory=False
                        ),

                        "agent_trainer_acessories" : ComponentListParameterSignature(mandatory=False, description="Acessories used for when the agent is learning"),

                       "limit_steps" : ParameterSignature(
                           default_value=-1,
                           description="Limits the steps in a single training session"
                          ),                         

                    "is_saving_in_memory" : ParameterSignature(default_value=True)

                       }
    
    exposed_values = {
                      "total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0,
                      "optimizations_done" : 0,
                      "average_optimization" : 0,
                      "external_end_requests" : None,
                      "is_training" : False,
                       "is_saving_in_memory" : True
                      } #this means we'll have a dic "values" with this starting values
    
    STATIC_EVENTS = {
        "ended_agent_training" : Event()
        }
    
    results_columns = ["episode", "episode_reward", "episode_steps", "avg_reward", "is_training"]
    
    def __init__(self, *args, **kwargs): #Initialization done only when the object is instantiated
        super().__init__(*args, **kwargs)

        self.values["external_end_requests"] = {} # this is to be sure that same instances are not shared TODO: Acessories could change       

    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.optimization_interval = self.get_input_value("optimization_interval")
        self.device = self.get_input_value("device")
    
        self.BATCH_SIZE = self.get_input_value("batch_size") #the number of transitions sampled from the replay buffer
        
        self.learn_with_all_memory = self.get_input_value("learn_with_all_memory")
        
        self.discount_factor = self.get_input_value("discount_factor") # the discount factor, A value of 0 makes the agent consider only immediate rewards, while a value close to 1 encourages it to look far into the future for rewards.
                              
        self.times_to_learn = self.get_input_value("times_to_learn")

        self.limit_steps = self.get_input_value("limit_steps")

        if self.limit_steps >= 0:
            self.lg.writeLine(f"Will limit the learning of this agent to {self.limit_steps} steps in this training session")    
        
        self._initialize_delays()    
                                
        self.initialize_agent()
        self.initialize_learner()
        self.initialize_memory()
        self.initialize_temp()
        self._initialize_acessories()

        self.values['is_training'] = False
        

    # INITIALIZATION ---------------------------------------------

    def _initialize_acessories(self):
        self.agent_trainer_acessories : list[AcessoryComponent] = self.get_input_value("agent_trainer_acessories")

        if self.agent_trainer_acessories is None:
            self.agent_trainer_acessories : list[AcessoryComponent] = []

        else:
            for acessory in self.agent_trainer_acessories:
                acessory.pass_input({"affected_component" : self})
                self.lg.writeLine(f"Agent trainer has acessory: {acessory.name}")

    def _initialize_delays(self):
        
        self.learning_start_ep_delay = self.get_input_value("learning_start_ep_delay")
        self.learning_start_step_delay = self.get_input_value("learning_start_step_delay")
        
    
    def initialize_agent(self):
    
        self.agent : AgentSchema = self.get_input_value("agent", look_in_attribute_with_name="agent")
        self.agent.process_input_if_not_processed()

        self.agent_policy : Policy = self.agent.policy
        
        
    def initialize_learner(self):
        
        self.learner : LearnerSchema = self.get_input_value("learner", look_in_attribute_with_name="learner")
        self.learner.pass_input({"device" : self.device, "agent" : self.agent, "agent_trainer" : self})
        


    def initialize_memory(self):
        
        self.memory : MemoryComponent = self.get_input_value("memory", look_in_attribute_with_name="memory")
        
        self.memory_fields_shapes = [] # tuples of (name_of_field, dimension)
            
        self.memory.pass_input({
                                    "device" : self.device,
                                })
        
        self.memory_transformer : MemorySampler = self.get_input_value("memory_transformer")

        if self.memory_transformer is not None:
            self.lg.writeLine(f"Will use memory transformer: {self.memory_transformer.name}")
        

    def initialize_temp(self):
        self.observation_memory_temp = self.agent.allocate_tensor_for_state()
        

        
    # RESULTS LOGGING --------------------------------------------------------------------------------
    
    @requires_input_process
    def calculate_results(self):
        
        return {
            "episode" : [self.values["episodes_done"]],
            "episode_reward" : [float(self.values["episode_score"])],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [float(self.values["episode_score"]) / self.values["episode_steps"]] if self.values["episode_steps"] > 0 else [0.0],
            "is_training" : [self.values['is_training']]
            }
        

    # PREDICTING OPTIMIZATIONS ------------------------------------------------

    @requires_input_process
    def make_optimization_prediction_for_agent_steps(self, total_steps):
        
        times_to_optimize_following_interval =  total_steps / self.optimization_interval
        to_return = 0
        
        if self.learn_with_all_memory:

            memory_capacity = self.memory.get_capacity()
            memory_ocupied = self.optimization_interval
            n_times_summed = 0

            # for each time we'll optimize without full memory
            while memory_ocupied < memory_capacity and n_times_summed < times_to_optimize_following_interval:

                times_to_learn_with_memory = int(memory_ocupied / self.BATCH_SIZE)
                to_return += times_to_learn_with_memory
                n_times_summed += 1
                memory_ocupied += self.optimization_interval

            # number of times we still have not processed
            number_of_times_still_to_learn = times_to_optimize_following_interval - n_times_summed

            # we process the times the optimization happens with full memory
            to_return += int(memory_capacity / self.BATCH_SIZE) * number_of_times_still_to_learn
        
        else:
            to_return = times_to_optimize_following_interval


        to_return = to_return * self.times_to_learn

        return to_return
    
    
    # STOP CONDITIONS ------------------------

    def _check_if_to_end_training_by_steps(self):

        if self.limit_steps >= 1: # if we're using steps to stop training

            if self.values["steps_done_in_session"] >= self.limit_steps:
                self.lg._writeLine(f"Total episodes done in this session, {self.values['steps_done_in_session']}, is greater than the limit for it, {self.limit_steps}")
                return True                
                
        return False

    def _check_if_to_end_training(self):
        return self._check_if_to_end_training_by_steps()
    
    # TRAINING_PROCESS ----------------------


    def is_agent_saving_in_memory(self):
        return self.values["is_saving_in_memory"]
    
    def make_agent_save_in_memory(self):
        self.values["is_saving_in_memory"] = True

    def make_agent_stop_saving_in_memory(self):
        self.values["is_saving_in_memory"] = False

        
    def is_agent_training(self):
        return self.values['is_training']
        
        
    @requires_input_process
    def setup_training_session(self):
        
        self.lg.writeLine("Setting up training session...\n")

        self.values['is_training'] = True

        external_end_requests : dict = self.values["external_end_requests"]
        
        for external_end_key in external_end_requests:
            self.initialize_external_end_request(external_end_key)

        self.values["steps_done_in_session"] = 0

        self.start_algorithm() # this tells the compon
        
        
                
    @requires_input_process
    def end_training(self):

        if self.values['is_training']:
            self.lg.writeLine("Ending training session... (Note that the trainer can still be used)\n")

            self.values['is_training'] = False

            self.EVENTS["ended_agent_training"].notify(self.name)

            self.end_algorithm()
        
    
    def _is_over(self):
        
        isover = super()._is_over()

        # the training is considered over when the reason for the training ended was external
        if not isover:
            if len(self.values["external_end_requests"]) > 0 and all(self.values["external_end_requests"].values()):
                self.lg.writeLine(f"Training is considered over because all external conditions asked for the training of the agent to end")
                isover = True

        return isover
        
    
    @requires_input_process
    def setup_episode(self, env : AECEnvironmentComponent):
        
        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0

        # this are optinal attributes that may be used to note a transition
        self._pending_prev_state = None
        self._pending_next_state = None
        self._pending_action = None
        self._has_pending_transition = False
        
        self.agent.reset_agent_in_environment(env.observe(self.agent.name))

        
            
        
    @requires_input_process
    def end_episode(self, i_episode=None, env: AECEnvironmentComponent=None):

        if env is not None and self._has_pending_transition:
            observation, reward, done, truncated, info = env.last()
            self.observe_new_state(env, observation)

            self._flush_pending_transition(
                i_episode=i_episode,
                env=env,
                reward=reward,
                done=done,
                truncated=truncated,
            )
        
        self.values["episodes_done"] = self.values["episodes_done"] + 1
        self.calculate_and_log_results()

        # we just log results when the agent is training
        if self.values['is_training']: 

            for acessory in self.agent_trainer_acessories:
                acessory.as_fun()

    
    # LEARNING AND INTERACTING WITH THE ENVIRONMENT -------------------------------
                
    def _clear_pending_transition(self):

        self._has_pending_transition = False
        self._pending_prev_state = None
        self._pending_next_state = None
        self._pending_action = None

    @requires_input_process
    def _flush_pending_transition(self, i_episode, env: AECEnvironmentComponent, reward, done, truncated):
        """
        Finalizes the pending transition for the last action taken by this agent.
        Use this when the agent gets another turn normally, or when the episode ends
        before that next turn happens.
        """
        if not self._has_pending_transition:
            return

        elif not self.values["is_saving_in_memory"]:
            self._clear_pending_transition()
            return

        next_state = self.agent.get_current_state_in_memory()

        self.do_after_training_step(
            i_episode=i_episode,
            action=self._pending_action,
            prev_state=None,   # do_after_training_step / observe_transition already uses pending prev_state path
            next_state=next_state,
            reward=reward,
            done=done,
            truncated=truncated,
        )

        self._clear_pending_transition()
        
    @requires_input_process
    def do_training_step(self, i_episode, env : AECEnvironmentComponent):
        
        '''
        Does a step, in which the agent acts and observers the transition
        Note that any other agents will not notice this change without outside coordination
            
        This method can be used when the agent is not training

        Note this method is not used in Parallel players or trainers, only aec
        '''

        observation, reward, done, truncated, info = env.last() # in aec, we receive the results of the previous episode step
        
        self.observe_new_state(env, observation)

        if self._has_pending_transition:

            self._flush_pending_transition(
                i_episode,
                env,
                reward,
                done,
                truncated
            )


        if done or truncated:
            env.step(None)
            return reward, done, truncated

        with torch.no_grad():                
            action = self.select_action_with_memory().squeeze(0) # decides the next action to take (can be random)

        self._pending_action = action
        self._has_pending_transition = True

        self._pending_prev_state = {**self.agent.state_memory}
        self._pending_prev_state["observation"] = self._pending_prev_state["observation"].detach().clone()

        env.step(action)

        # we don't use pending next state, as the state will be the one computed after the other agent acts
                
        return reward, done, truncated
             
             
                                         
    def do_after_training_step(self, i_episode=None, action=None, prev_state=None, next_state=None, reward=None, done=None, truncated=None):

        '''
        Does the normal computation after a training step
        This assumes an observation was just made, an action chosen and the environment was notified of that action
        
        It observeses the transition of the environment to the received observation, and notes that an episode step was just made
        '''

        self.values["episode_score"] = self.values["episode_score"] + reward

        if self.values["is_saving_in_memory"]:
                            
            self.observe_transition_to(prev_state=prev_state, 
                                        new_state=next_state,
                                        action=action,
                                        reward=reward,
                                        done=done,
                                        truncated=truncated)
            
        self.values["episode_steps"] = self.values["episode_steps"] + 1
        self.values["total_steps"] = self.values["total_steps"] + 1 #we just did a step      ~

        if self.values['is_training']:

            self.values["steps_done_in_session"] = self.values["steps_done_in_session"] + 1
            
            if self._check_if_to_end_training():
                self.end_training()

        self._learn_if_needed() # uses the learning strategy to learn if it verifies the conditions to do so
                
        

    def observe_transition_to(self, prev_state=None, new_state=None, action=None, reward=None, done=None, truncated=None, **kwargs):
        
        '''
        Makes agent observe and remember a transiction from its (current) a state to another
        This does not affect the agent short term memory, it targets the memory of transitions that will use the training
        '''        

        if reward is None or action is None or done is None or truncated is None:
            raise NotImplementedError(f"Reward, action, done and truncated must always be passed to observe a transition")

        if prev_state is None and new_state is None and self._pending_next_state is None and self._pending_prev_state is None:
            raise Exception(f"Either prev state must be defined or new_state must be defined")


        if new_state is not None: # if we received a new state
            new_state = self.agent.get_internal_agent_state_with_new(new_state).clone()

        elif self._pending_next_state is not None: # if we have a pending next state
            new_state = self._pending_next_state

        else:
            new_state = self.agent.get_current_state_in_memory()



        if prev_state is not None:
            pass

        elif self._pending_prev_state is not None:
            prev_state = self._pending_prev_state

        else:
            prev_state = self.agent.get_current_state_in_memory()
        
                        
        return self._observe_transition_to(prev_state, new_state, action, reward, done, truncated, **kwargs)
    

    def _observe_transition_to(self, prev_state, new_state, action, reward, done, truncated, **kwargs):
        pass
    
    def push_to_memory(self, to_push):
        self.memory.push(to_push) 

        
        
    def observe_new_state(self, env : AECEnvironmentComponent, new_state=None):
        '''
        Makes the agent observe a new state, remembering it in case it needs that information in future computations
        This is not a stored transition in the memory of the trainer
        '''

        if new_state is None:
            new_state = env.observe(self.agent.name)

        self.agent.update_state_memory(new_state)
        

    def select_action(self, state):
        
        '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''

        return self.agent.policy_predict(state)
    

    def select_action_with_memory(self):
        
        return self.agent.policy_predict_with_memory()

    # LEARNING PROCESS --------------------------------------------


    def _learn_if_needed(self):

        '''
        Does a learning step if the agent is still learning
        This means calling optimize agent          
        '''

        if self.values['is_training']:
        
            can_learn_by_ep_delay = self.learning_start_ep_delay < 1 or self.values["episodes_done"] >= self.learning_start_ep_delay
            can_learn_by_step_delay = self.learning_start_step_delay < 1 or self.values["total_steps"] >= self.learning_start_step_delay

            if can_learn_by_ep_delay and can_learn_by_step_delay:          
            
                if self.values["total_steps"] % self.optimization_interval == 0:

                    self.optimizeAgent()


    def optimizeAgent(self):

        '''Optimizes the trained agent for the number of specified times''' 

        if self.memory_transformer is not None:
            self.memory_transformer.prepare(self.memory)

        for _ in range(self.times_to_learn):
            self.do_learning_with_memory_learner() 
           

        if self.memory_transformer is not None:
            self.memory_transformer.let_go()
        
        
    def _optimize_policy_model_with_batch(self, batch):
        self.learner.learn(batch)

        

    def do_learning_with_memory_learner(self):

        sampler = self.memory if self.memory_transformer is None else self.memory_transformer

        if self.BATCH_SIZE != None:
        
            if len(self.memory) < self.BATCH_SIZE:
                return

            if self.learn_with_all_memory:
                batches = sampler.sample_all_with_batches(self. BATCH_SIZE)
            
            else:
                batches = [sampler.sample(self.BATCH_SIZE)]

        else:
            batches = [sampler.get_all()]        
                
        for b in batches:
            self._optimize_policy_model_with_batch(b)

        if len(batches) > 0:
             self.values["optimizations_done"] += 1


    # EXTERNAL ACESSORIES PROCESSING ----------------------------------------------

    def request_end_from_external(self, key):

        '''
        Receives a request from an outside entity to terminate the training of this agent
        This usually means convergence was detected
        '''

        external_end_requests : dict = self.values["external_end_requests"]
        external_end_requests[key] = True

        if all(external_end_requests.values()):
            self.lg.writeLine(f"All external end requests: {external_end_requests} are true, requesting to end training...")
            self.end_training()

    def request_continue_from_external(self, key):

        external_end_requests : dict = self.values["external_end_requests"]
        external_end_requests[key] = False


    def initialize_external_end_request(self, key):

        self.lg.writeLine(f"Received request to register an end condition with key: {key}")

        external_end_requests = self.values["external_end_requests"]
        external_end_requests[key] = False