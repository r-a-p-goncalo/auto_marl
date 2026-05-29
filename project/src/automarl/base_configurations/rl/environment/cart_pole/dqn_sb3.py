



'''
This is based on the stable baselines 3 hyperparameter configuration, in:

https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

With hyperparameters:

# Almost Tuned
CartPole-v1:
  n_timesteps: !!float 5e4
  policy: 'MlpPolicy'
  learning_rate: !!float 2.3e-3
  batch_size: 64
  buffer_size: 100000
  learning_starts: 1000
  gamma: 0.99
  target_update_interval: 10
  train_freq: 256
  gradient_steps: 128
  epsilon_decay: 0.16
  exploration_final_eps: 0.04
  policy_kwargs: "dict(net_arch=[256, 256])"


'''

from automarl.components.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.rl.environment.gymnasium.aec_gymnasium_env import AECGymnasiumEnvironmentWrapper
from automarl.components.rl.exploration.epsilong_greedy import EpsilonGreedyLinearStrategy, EpsilonGreedyStrategy
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema
from automarl.components.rl.policy.qpolicy import QPolicy
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component_dqn import AgentTrainerDQN
from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.rl.trainers.rl_trainer.single_rl_trainer import SingleRLTrainer


def config_dict():
    

    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": (AECGymnasiumEnvironmentWrapper, {"environment" : "CartPole-v1"}),
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
                ),
            "state_translator" : (ToTorchTranslator, {})
        },
        
        "rl_trainer" : (SingleRLTrainer,
            
            {
            "limit_total_steps" : 5e4,
            "default_trainer_class" : AgentTrainerDQN,
            "agents_trainers_input" : { #for each agent trainer
                
                "optimization_interval" : 256,
                
                "discount_factor" : 0.99,
                
                "learning_start_step_delay" : 1000,
                
                "batch_size" : 64,
                
                "times_to_learn" : 128, # how many times to optimize at learning time
                
                "learner" : (DeepQLearnerSchema, {
                                "target_update_learn_interval" : 5, # how many optimization steps before updating the target model
                               "target_update_rate" : 1.0, # the target model is totally replaced
                               "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "learning_rate" : 2.3e-3
                                   }
                )
                }),
            
                "memory" : (TorchMemoryComponent, {
                    "capacity" : 100000
                }),
                
                "exploration_strategy" : (EpsilonGreedyLinearStrategy,
                                                                  {
                                            "epsilon_end" : 0.04,
                                            "epsilon_start" : 1.0,
                                            "epsilon_decay" :  0.16
                                                                  }
                                          )
                
            
            }

            }
        )
        
    }
}