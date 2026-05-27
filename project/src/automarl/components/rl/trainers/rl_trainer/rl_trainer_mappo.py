from automarl.components.rl.trainers.rl_trainer.parallel_rl_trainer import RLTrainerComponentParallel
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.ml.memory.memory_samplers.advantages_calc_sampler import PPOAdvantagesCalcSampler
from automarl.components.rl.trainers.rl_trainer.rl_trainer_orquestrator import RLTrainerOrquestrator
from automarl.components.rl.learners.ppo_learner_separated import PPOLearnerOnlyCritic
from automarl.core.input_management import ParameterSignature
from automarl.components.rl.trainers.agent_trainer_ppo import AgentTrainerPPOCriticAware

import torch


class RLTrainerMAPPO(RLTrainerOrquestrator, RLTrainerComponentParallel):

    parameters_signature = {
        "default_trainer_class": ParameterSignature(default_value=AgentTrainerPPOCriticAware),

        "critic_learner": ComponentParameterSignature(
            default_component_definition=(PPOLearnerOnlyCritic, {})
        ),

        "memory_transformer": ComponentParameterSignature(
            default_component_definition=(PPOAdvantagesCalcSampler, {})
        ),

        "times_to_learn": ParameterSignature(
            default_value=1,
            description="How many times to optimize the critic at learning time",
            custom_dict={"hyperparameter_suggestion": ["int", {"low": 1, "high": 20}]}
        ),

        "learn_with_all_memory": ParameterSignature(
            default_value=False,
            description="When true, each learning will consist of dividing the entire memory into batches with the specified size, and learning with each of them"
        ),

        "batch_size": ParameterSignature(
            mandatory=False,
            custom_dict={"hyperparameter_suggestion": ["cat", {"choices": [8, 16, 32, 64, 128, 256]}]}
        ),

        "discount_factor": ParameterSignature(get_from_parent=True),
    }

    def _process_input_internal(self):
        super()._process_input_internal()

        self.lg.writeLine("Processing mappo specific input...")

        self.BATCH_SIZE = self.get_input_value("batch_size")
        self.learn_with_all_memory = self.get_input_value("learn_with_all_memory")
        self.times_to_learn = self.get_input_value("times_to_learn")

        self._initialize_critic_learner()

        self.memory_transformer.pass_input({"learner": self.critic_learner})

        self._critic_optimizations_prediction()

        self.lg.writeLine("Finished processing mappo specific input\n")

    def _initialize_agent_indexing(self):

        self.all_agents = self.values.get("all_agents", None)
        if self.all_agents is None:
            self.all_agents = list(self.env.agents())
            self.values["all_agents"] =self.all_agents

        self.num_agents = len(self.all_agents)
        self.agent_to_idx = {agent_name: i for i, agent_name in enumerate(self.all_agents)}

        self.lg.writeLine(f"MAPPO fixed agent ordering: {self.all_agents}")

    def setup_agents(self):
        super().setup_agents()

        self.lg.writeLine(
            "As MAPPO will coordinate transitions stored to memory, will turn off automatic saving to memory"
        )

        for agent_trainer in self.agents_trainers.values():
            agent_trainer.make_agent_stop_saving_in_memory()

    def _initialize_critic_learner(self):
        self.critic_learner: PPOLearnerOnlyCritic = self.get_input_value("critic_learner")

        critic_model_input = self.critic_learner.get_input_value("critic_model_input")
        critic_model_input = {} if critic_model_input is None else critic_model_input
        critic_model_input = {
            **critic_model_input,
            "input_shape": self.whole_observation_shape,
            "output_shape": self.num_agents
        }

        self.critic_learner.pass_input({
            "critic_model_input": critic_model_input,
            "num_agents": self.num_agents,
        })

    def initialize_memory(self):
        self._initialize_agent_indexing()

        super().initialize_memory()

        state_shape = {**self.env.get_whole_state_shape()}
        self.lg.writeLine(f"Whole state shape is {state_shape}")

        self.whole_observation_shape = state_shape.pop("observation")

        self.memory_fields_shapes = [
            *self.memory_fields_shapes,
            ("observation", self.whole_observation_shape),
            ("next_observation", self.whole_observation_shape),
            ("reward", self.num_agents),
            ("done", self.num_agents),
            ("truncation", self.num_agents),
            ("observation_old_critic_value", self.num_agents),
            ("next_obs_old_critic_value", self.num_agents),
            ("alive_agents", self.num_agents),
        ]

        self.memory.pass_input({
            "device": self.device,
            "transition_data": self.memory_fields_shapes
        })

    def _build_agent_vector(self, values_dict, default_value=0.0, dtype=torch.float32):
        vec = torch.full((self.num_agents,), default_value, dtype=dtype, device=self.device)

        for agent_name, value in values_dict.items():
            idx = self.agent_to_idx[agent_name]
            vec[idx] = value

        return vec

    def _build_alive_agents_vector(self, active_agent_names):
        alive_vec = torch.zeros((self.num_agents,), dtype=torch.float32, device=self.device)

        for agent_name in active_agent_names:
            idx = self.agent_to_idx[agent_name]
            alive_vec[idx] = 1.0

        return alive_vec

    def _push_shared_transition(
        self,
        prev_whole_state,
        next_whole_state,
        observations,
        rewards,
        actions,
        dones,
        truncations,
        agent_names
    ):
        prev_whole_obs = torch.tensor(prev_whole_state["observation"], dtype=torch.float32, device=self.device)
        next_whole_obs = torch.tensor(next_whole_state["observation"], dtype=torch.float32, device=self.device)

        observation_critic_value = self.critic_learner.critic_pred(prev_whole_obs).reshape(-1)
        next_observation_critic_value = self.critic_learner.critic_pred(next_whole_obs).reshape(-1)

        reward_vec = self._build_agent_vector(rewards, default_value=0.0, dtype=torch.float32)
        done_vec = self._build_agent_vector(dones, default_value=1.0, dtype=torch.float32)
        trunc_vec = self._build_agent_vector(truncations, default_value=1.0, dtype=torch.float32)
        alive_agents_vec = self._build_alive_agents_vector(agent_names)

        transition = {
            "observation": prev_whole_obs,
            "next_observation": next_whole_obs,
            "reward": reward_vec,
            "done": done_vec,
            "truncation" : trunc_vec,
            "observation_old_critic_value": observation_critic_value.detach(),
            "next_obs_old_critic_value": next_observation_critic_value.detach(),
            "alive_agents": alive_agents_vec.detach(),
        }

        self.memory.push(transition)

        for agent_name in agent_names:
            idx = self.agent_to_idx[agent_name]
            agent_trainer = self.agents_trainers[agent_name]

            agent_trainer.observe_transition_to(
                new_state=torch.tensor(
                    observations[agent_name]["observation"],
                    dtype=torch.float32,
                    device=self.device
                ),
                action=actions[agent_name],
                reward=rewards[agent_name],
                done=dones[agent_name],
                prev_critic_val=observation_critic_value[idx],
                next_critic_val=next_observation_critic_value[idx],
                truncated=truncations[agent_name]
            )

    def run_single_episode(self, i_episode):
        self.setup_single_episode(i_episode)

        while True:
            agent_names = [*self.env.get_active_agents()]

            if len(agent_names) == 0:
                break

            prev_whole_state = self.env.get_current_whole_state()
            actions = self.choose_actions_for_agents(agent_names, i_episode)

            observations, rewards, terminations, truncations, infos = self.env.step(actions)
            next_whole_state = self.env.get_current_whole_state()

            done = self.process_env_step_for_agents(
                i_episode, agent_names, actions, observations, rewards, terminations, truncations
            )

            self._push_shared_transition(
                prev_whole_state=prev_whole_state,
                next_whole_state=next_whole_state,
                observations=observations,
                rewards=rewards,
                actions=actions,
                dones=terminations,
                truncations=truncations,
                agent_names=agent_names
            )

            team_reward = self._aggregated_reward(rewards)
            self.after_environment_step(team_reward)

            if done or self._check_if_to_end_episode():
                break

        for agent_in_training in self.agents_trainers.values():
            agent_in_training.end_episode(env=self.env, i_episode=i_episode)

        self.values["episodes_done"] += 1
        self.values["episodes_done_in_session"] += 1

        self.calculate_and_log_results()

    def _optimize_critic(self):
        if self.BATCH_SIZE is not None:
            if len(self.memory) < self.BATCH_SIZE:
                return

            if self.learn_with_all_memory:
                critic_batches = self.memory_transformer.sample_all_with_batches(self.BATCH_SIZE)
            else:
                critic_batches = [self.memory_transformer.sample(self.BATCH_SIZE)]
        else:
            critic_batches = [self.memory_transformer.get_all()]

        for batch in critic_batches:
            self.critic_learner.learn(batch)

    def _pre_agents_optimization(self):
        super()._pre_agents_optimization()
        for _ in range(self.times_to_learn):
            self._optimize_critic()

    def _pos_agents_optimization(self):
        super()._pos_agents_optimization()

    def optimize_agents(self):
        super().optimize_agents()
        self.memory.clear()


    def _critic_optimizations_prediction(self):

        '''Setup the predicted value for the optimizations to do'''

        if self.predict_optimizations_to_do:

            self.lg._writeLine("RLTrainer will try to predict the optimizations for the critic")

            if self.num_episodes > 1 and self.limit_total_steps > 1:
               raise Exception("Can't make prediction")
    
            elif self.num_episodes > 1:
                raise NotImplementedError("Critic optimization prediction by episodes is not implemented for MAPPO")
        
            elif self.limit_total_steps > 1:
             
                times_to_optimize_following_interval = self.limit_total_steps / self.optimization_interval
                optimizations_for_critic = 0
        
                if self.BATCH_SIZE is None:
                    optimizations_for_critic = times_to_optimize_following_interval
        
                elif self.learn_with_all_memory:
                 
                    memory_capacity = self.memory.get_capacity()
                    memory_ocupied = self.optimization_interval
                    n_times_summed = 0
        
                    # before memory is full
                    while memory_ocupied < memory_capacity and n_times_summed < times_to_optimize_following_interval:
                     
                        times_to_learn_with_memory = int(memory_ocupied / self.BATCH_SIZE)
                        optimizations_for_critic += times_to_learn_with_memory
                        n_times_summed += 1
                        memory_ocupied += self.optimization_interval
        
                    # after memory is full
                    number_of_times_still_to_learn = times_to_optimize_following_interval - n_times_summed
                    optimizations_for_critic += int(memory_capacity / self.BATCH_SIZE) * number_of_times_still_to_learn
        
                else:
                    optimizations_for_critic = times_to_optimize_following_interval
        
                optimizations_for_critic = int(optimizations_for_critic * self.times_to_learn)
        
                self.lg._writeLine(
                    f"RLTrainer predicted it will do {optimizations_for_critic} optimizations for critic"
                )
        
                self.critic_learner.values["optimizations_to_do"] = optimizations_for_critic
        
            else:
                raise Exception("Can't make prediction")