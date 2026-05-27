from automarl.components.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema
from automarl.components.rl.policy.qpolicy import QPolicy

from automarl.components.rl.environment.pettingzoo.parallel_petting_zoo_env import PettingZooEnvironmentWrapperParallel
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer_component_dqn import AgentTrainerDQN
from automarl.components.rl.trainers.rl_trainer_component import RLTrainerComponent

DEFAULT_ENV_DEFINITION = (PettingZooEnvironmentWrapperParallel, {
    "environment": "cooperative_pong"})


def config_dict(num_episodes=200, env=None):

    if env is None:
        env = DEFAULT_ENV_DEFINITION

    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": env,
        "agents_input": {
            "state_memory_size" : 2,

            "policy" : ( QPolicy,
                        {
                        "model" : (
                            FullyConnectedModelSchema, 
                            {
                            "layers" : [64, 64]
                            }
                            ),
                        }
                )
        },
        
        "rl_trainer" : (RLTrainerComponent,
            
            {
            "num_episodes" : num_episodes,
            "optimization_interval": 300,
            "default_trainer_class" : AgentTrainerDQN,
            "agents_trainers_input" : { #for each agent trainer
                
                "learner" : (DeepQLearnerSchema, {
                               "target_update_rate" : 0.05,
                               "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "learning_rate" : 0.0001
                                   }
                )
                }),
            
                "memory" : (TorchDiskMemoryComponent, {
                    "capacity" : 500
                }),
                
                "exploration_strategy" : (EpsilonGreedyStrategy,
                                                                  {
                                            "epsilon_end" : 0.1,
                                            "epsilon_start" : 0.99,
                                            "epsilon_decay" : 0.99
                                                                  }
                                          )
                
            
            }

            }
        )
        
    }
}
    
def mockup_config_dict(num_episodes=200):
    
    raise NotImplementedError("This is not up to date")
    
    from automarl.components.ml.models.mockups.model_mockup import MockupRandomModel
    from automarl.components.ml.optimizers.mockups.mockup_optimizers import MockupOptimizerSchema

    return {
    
    "__type__": "<class 'automarl.components.rl.rl_pipeline.RLPipelineComponent'>",
    "name": "RLPipelineComponent",
    "input": {
        "num_episodes_per_run": num_episodes,
        "optimization_interval": 300,
        "device" : "cpu",
        "environment": {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader"        
        },
        "agents_input": {
            "exploration_strategy_class" : str(EpsilonGreedyStrategy),
            "state_memory_size" : 2,
            "exploration_strategy_input" : {
                "epsilon_end" : 0.1,
                "epsilon_start" : 0.99,
                "epsilon_decay" : 0.99
                },
            "policy" : ( QPolicy,
                        {
                        "model" : ( MockupRandomModel,{})
                        }
                ),
            "learner" : (DeepQLearnerSchema, {
                "target_update_rate" : 0.05,
                "optimizer" :( MockupOptimizerSchema, {})
            }),
            "memory_input" : {
                "capacity" : 300
            }
        },

    },
    "child_components": [
        {
            "__type__": str(PettingZooEnvironmentWrapper),
            "name": "PettingZooEnvironmentLoader",
            "input": {
                "environment": "cooperative_pong"
            }
        }
    ]
}