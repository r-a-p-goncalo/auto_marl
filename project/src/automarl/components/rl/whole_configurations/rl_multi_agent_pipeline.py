from automarl.components.ml.memory.torch_disk_memory_component import TorchDiskMemoryComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.conv_model import ConvModel
from automarl.components.ml.models.dynamic_conv_model import DynamicConvModelSchema
from automarl.components.ml.models.joint_model import ModelSequenceComponent
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema
from automarl.components.rl.policy.qpolicy import QPolicy

from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer_component_dqn import AgentTrainerDQN
from automarl.components.rl.trainers.rl_trainer_component import RLTrainerComponent
from automarl.components.rl.environment.pettingzoo.parallel_petting_zoo_env import PettingZooEnvironmentWrapperParallel
from automarl.components.rl.trainers.rl_trainer.parallel_rl_trainer import RLTrainerComponentParallel
from automarl.components.fundamentals.translator.torch_image_state_translator import ImageReverterToSingleChannel, ImageNormalizer
from automarl.components.fundamentals.translator.translator import TranslatorSequence
from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator


def config_dict():


    return {
    
    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",
    "input": {
        "device" : "cpu",
        "environment": (PettingZooEnvironmentWrapperParallel, {
                            "environment": "cooperative_pong"}),
        "agents_input": {
            "state_memory_size" : 2,

            "state_translator" : (TranslatorSequence, {
                "translators_sequence" : [
                        (ToTorchTranslator, {}),
                        (ImageReverterToSingleChannel, {}),
                        (ImageNormalizer, {})
                    ],
                    "in_place_translation" : True
                },
            
            ),

            "policy" : ( QPolicy,
                        {

                        "model" : (
                            ModelSequenceComponent,
                             {"models": [ 

                                [("__get_by_name__", {"name_of_component" : "shared_model"})], 
                                
                                (FullyConnectedModelSchema, 
                                {
                                "device" : "cuda",
                                "layers" : [64, 32]
                                }
                                ),
                             ]}

                        )
                        }
            )
        },
        
        "rl_trainer" : (RLTrainerComponentParallel,
            
            {
            "limit_total_steps" : 10e4,
            
            "default_trainer_class" : AgentTrainerDQN,
            "agents_trainers_input" : { #for each agent trainer
                
                "optimization_interval": 1000,
                "times_to_learn" : 64,
                "batch_size" : 64,

                "learner" : (DeepQLearnerSchema, {
                                "device" : "cuda",
                               "target_update_rate" : 1.0,
                               "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "learning_rate" : 0.00001
                                   }
                )
                }),
            
                "memory" : (TorchMemoryComponent, {
                    "device" : "cpu",
                    "capacity" : 1000
                }),
                
                "exploration_strategy" : (EpsilonGreedyStrategy,
                                                                  {
                                            "epsilon_end" : 0.04,
                                            "epsilon_start" : 1.0,
                                            "epsilon_decay" : 0.16
                                                                  }
                                          )
    
                    }

            }
        )
        
    },
    "child_components" : [
        {
            "__type__" : DynamicConvModelSchema,
            "name" : "shared_model",
            "input" : {
                "cnn_layers" : [
                    {"kernel_size" : 8, "out_channels" : 32, "stride" : 4},
                    {"kernel_size" : 4, "out_channels" : 64, "stride" : 2},
                    {"kernel_size" : 3, "out_channels" : 64, "stride" : 1},
                    

                ]
            }
        }
    ]
}