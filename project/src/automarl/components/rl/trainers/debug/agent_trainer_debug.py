


from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
from automarl.components.rl.trainers.agent_trainer_component import AgentTrainer
from automarl.components.ml.models.torch_model_components import TorchModelComponent
from automarl.core.input_management import ParameterSignature
from automarl.components.ml.models.torch_model_utils import model_parameter_distance

class AgentTrainerDebug(AgentTrainer, ComponentDebug):
        
        '''A debug class that is not meant to be used '''

        is_debug_schema = True

        parameters_signature = {
             "verify_model_difference_after_optimize" : ParameterSignature(default_value=True),
             "note_observed_transitions" : ParameterSignature(default_value=True),
             "note_chosen_actions" : ParameterSignature(default_value=False)
        }

        def _process_input_internal(self):

            super()._process_input_internal()

            self.verify_model_difference_after_optimize = self.get_input_value("verify_model_difference_after_optimize")
    
            if self.verify_model_difference_after_optimize:
                self.model : TorchModelComponent = self.agent_policy.model
                self.model.process_input_if_not_processed()

                self.lg.writeLine(f"Creating temporary model to note difference in optimizations...")

                self.__temporary_model : TorchModelComponent = self.model.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_comp_opti", "create_new_directory" : False}, is_deep_clone=True)
        
            self.note_observed_transitions = self.get_input_value("note_observed_transitions")
            
            if self.note_observed_transitions:
                self.lg.writeLine(f"total_step, episode, episode_step: state + action -> new_state, reward, done\n", file="observed_transitions.txt", use_time_stamp=False)
            
            self.lg.writeLine(f"total_step, episode, episode_step: reward, done\n", file="training_steps.txt", use_time_stamp=False)
    
            self.note_chosen_actions = self.get_input_value("note_chosen_actions")

            if self.note_chosen_actions:
                self.lg.writeLine(f"state -> action\n", file="chosen_actions.txt", use_time_stamp=False)

    
        def do_training_step(self, i_episode, env):
        
            reward, done, truncated = super().do_training_step(i_episode, env)

            # we could do something here

            return reward, done, truncated
        
        
        def do_after_training_step(self, i_episode=None, action=None, prev_state=None, next_state=None, reward=None, done=None, truncated=None):
             
            super().do_after_training_step(i_episode, action, prev_state, next_state, reward, done, truncated)

            self.lg.writeLine(f"After training step: {self.values['total_steps']}, {self.values['episodes_done']}, {self.values['episode_steps']}: {reward}, {done}", file="training_steps.txt")

        def optimizeAgent(self):
            
            self.lg.writeLine(f"In episode (total) {self.values['episodes_done']}, optimizing at step {self.values['episode_steps']} that is the total step {self.values['total_steps']}", file=self.TRAIN_LOG)

            return super().optimizeAgent()


        def do_learning_with_memory_learner(self):
        
            if self.verify_model_difference_after_optimize:
                self.__temporary_model.clone_other_model_into_this(self.model)
    
            super().do_learning_with_memory_learner()
    
            if self.verify_model_difference_after_optimize:
                l2_distance, avg_distance, cosine_sim = model_parameter_distance(self.__temporary_model, self.model)

                self.lg.writeLine(f"Optimized agent, clone has id {id(self.__temporary_model.model)} and optimized_model {id(self.model.model)}:", file="model_optimization.txt")
                self.lg.writeLine(f"    l2_distance: {l2_distance}\n    avg_distance: {avg_distance}\n    cosine_sime: {cosine_sim}\n", file="model_optimization.txt")
    

    
        def _observe_transition_to(self, prev_state, new_state, action, reward, done, truncated):

            if not self.note_observed_transitions:
                super()._observe_transition_to(prev_state, new_state, action, reward, done, truncated)

            else:


                _old_state_obs = prev_state["observation"].clone()
                _new_state_obs = new_state["observation"].clone()

                super()._observe_transition_to(prev_state, new_state, action, reward, done, truncated)


                old_state_str = str(_old_state_obs)
                if len(old_state_str) > 30:
                    old_state_str = f"{old_state_str[10:]}...{old_state_str[:10]}"   

                new_state_str = str(_new_state_obs)
                if len(new_state_str) > 30:
                    new_state_str = f"{new_state_str[10:]}...{new_state_str[:10]}"   

                self.lg.writeLine(f"{self.values['total_steps']}, {self.values['episodes_done']}, {self.values['episode_steps']}: {old_state_str} + {action} -> {new_state_str}, {reward}, {done}", file="observed_transitions.txt")
    

        def select_action(self, state):

            '''uses the exploration strategy defined, with the state, the agent and training information, to choose an action'''

            action = super().select_action(state)


            if self.note_chosen_actions:
                old_state_str = str(state)
                if len(old_state_str) > 30:
                    old_state_str = f"{old_state_str[10:]}...{old_state_str[:10]}"

                self.lg.writeLine(f"{old_state_str} -> {action}", file="chosen_actions.txt")

            return action


        def select_action_with_memory(self):
        
            action = super().select_action_with_memory()

            if self.note_chosen_actions:
                self.lg.writeLine(f"in_memory -> {action}", file="chosen_actions.txt")

            return action