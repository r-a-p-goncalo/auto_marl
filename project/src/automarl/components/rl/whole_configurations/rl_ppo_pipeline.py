from automarl.components.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema
from automarl.components.rl.policy.qpolicy import QPolicy
from automarl.components.rl.policy.stochastic_policy import StochasticPolicy
from automarl.components.rl.learners.ppo_learner import PPOLearner

from automarl.components.rl.environment.pettingzoo.pettingzoo_env import PettingZooEnvironmentWrapper
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer
from automarl.components.rl.trainers.agent_trainer.agent_trainer_ppo import AgentTrainerPPO
from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent


def config_dict(num_episodes=200):

    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cuda",
        "environment": {
            "__type__": PettingZooEnvironmentWrapper,
            "name": "PettingZooEnvironmentLoader"        
        },
        "agents_input": {
            "state_memory_size" : 2,

            "policy" : ( StochasticPolicy,
                        {
                        "model" : (
                            FullyConnectedModelSchema, 
                            {
                            "hidden_layers" : 3,
                            "hidden_size" : 64
                            }
                            ),
                        }
                )
        },
        
        "rl_trainer" : (RLTrainerComponent,
            
            {
            "default_trainer_class" : AgentTrainerPPO,
            "num_episodes" : num_episodes,
            "optimization_interval": 300,
            "agents_trainers_input" : { #for each agent trainer
                
                "learner" : (PPOLearner, {}),
            
                "memory" : (TorchDiskMemoryComponent, {
                    "capacity" : 500
                })
                
            
            }

            }
        )
        
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


