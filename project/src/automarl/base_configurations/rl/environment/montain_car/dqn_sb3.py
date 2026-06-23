



'''
This is based on the stable baselines 3 hyperparameter configuration, in:

https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

With hyperparameters:

MountainCar-v0:
  n_timesteps: !!float 1.2e5
  policy: 'MlpPolicy'
  learning_rate: !!float 4e-3
  batch_size: 128
  buffer_size: 10000
  learning_starts: 1000
  gamma: 0.98
  target_update_interval: 600
  train_freq: 16 # trains at an interval of 16 episodes
  gradient_steps: 8 # each time it trains, it does 8 episode steps
  epsilon_decay: 0.2
  exploration_final_eps: 0.07
  policy_kwargs: "dict(net_arch=[256, 256])"


'''

from automarl.components.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.rl.environment.gymnasium.gymnasium_env import GymnasiumEnvironmentWrapper
from automarl.components.rl.exploration.epsilong_greedy import EpsilonGreedyLinearStrategy, EpsilonGreedyStrategy
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema
from automarl.components.rl.policy.qpolicy import QPolicy
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component_dqn import AgentTrainerDQN
from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent

def experiment_name():
    return "montaincar_dqn_sb3"

def config_dict():


    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": (GymnasiumEnvironmentWrapper, {"environment" : "MountainCar-v0"}),
        "agents_input": {
            
            "policy" : ( QPolicy,
                        {
                        "model" : (
                            FullyConnectedModelSchema, 
                            {
                            "hidden_layers" : 2,
                            "hidden_size" : 256
                            }
                            ),
                        }
                )
        },
        
        "rl_trainer" : (RLTrainerComponent,
            
            {
            "limit_total_steps" : 1.2e5,
            "default_trainer_class" : AgentTrainerDQN,
            "agents_trainers_input" : { #for each agent trainer
                
                "optimization_interval" : 16,
                
                "discount_factor" : 0.98,
                
                "learning_start_ep_delay" : 1000,
                
                "batch_size" : 128,
                
                "times_to_learn" : 8,
                
                "learner" : (DeepQLearnerSchema, {
                                "target_update_learn_interval" : 5,
                               "target_update_rate" : 1.0, # the target model is totally replaced
                               "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "learning_rate" : 2.3e-3
                                   }
                )
                }),
            
                "memory" : (TorchMemoryComponent, {
                    "capacity" : 10000
                }),
                
                "exploration_strategy" : (EpsilonGreedyLinearStrategy,
                                                                  {
                                            "epsilon_end" : 0.07,
                                            "epsilon_start" : 1.0,
                                            "epsilon_decay" : 0.2
                                                                  }
                                          )
                
            
            }

            }
        )
        
    }
}