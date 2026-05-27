from automarl.components.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.dynamic_conv_model import DynamicConvModelSchema
from automarl.components.ml.models.joint_model import ModelSequenceComponent
from automarl.components.ml.models.model_initialization_strategy import TorchModelInitializationStrategyOrthogonal
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.rl.learners.ppo_learner import PPOLearner

from automarl.components.rl.policy.stochastic_policy import StochasticPolicy
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer_ppo import AgentTrainerPPO
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
        "do_full_setup_of_seed" : True,
        "device" : "cuda",
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
                }
            
            ),

            "policy" : ( StochasticPolicy,
                        {

                        "model" : (
                            ModelSequenceComponent,
                             {  "name" : "policy_model",
                                "models": [ 

                                [("__get_by_name__", {"name_of_component" : "shared_model"})], 
                                
                                (FullyConnectedModelSchema, 
                                {
                                "name" : "policy_head",
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

            "name": "RLTrainerComponent",
            "limit_total_steps" : 500000,
            "predict_optimizations_to_do" : True, 
            "default_trainer_class" : AgentTrainerPPO,
            "agents_trainers_input" : { #for each agent trainer
                
                "optimization_interval": 256,
                "times_to_learn" : 8,
                "batch_size" : 32,

                "learner" : (PPOLearner, {

                     "lambda_gae": 0.8,
                     "clip_epsilon" : 0.2,
                     "entropy_coef" : 0.01,
                     "value_loss_coef" : 0.5,

                    "critic_model" : (

                            ModelSequenceComponent,
                             {  "name" : "critic_model",
                                "output_shape" : 1,
                                "models": [ 

                                [("__get_by_name__", {"name_of_component" : "shared_model"})], 
                                
                                (FullyConnectedModelSchema, 
                                {
                                "name" : "critic_head",
                                "layers" : [64]
                                }
                                ),
                                 ]}
                            ),

                    "optimizer" :(
                                   AdamOptimizer,
                                   {
                                       "name" : "AdamOptimizerComponent",
                                       "learning_rate" : 2.3e-3,
                                       "linear_decay_learning_rate_with_final_input_value_of" : ("relative", [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "__any__"]})]),
                                       "clip_grad_value" : (
                                           DynamicLinearValueInRangeBasedOnComponent, {
                                               "input_for_fun_key" : "optimizations_done",
                                               "initial_value" : 0.2,
                                               "final_value" : 0,
                                               "input_component" : ('relative', ("__get_by_name__", {"name_of_component" : "AdamOptimizerComponent"})),
                                               "input_for_fun_max_value" : 
                                                ('relative', 
                                                 [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "__any__"]})]
                                                )


                                           }
                                       )                 
                                   }
                )
                }),
            
                "memory" : (TorchMemoryComponent, {
                    "device" : "cpu",
                    "capacity" : 256
                })
    
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
                    {"kernel_size" : 8, "out_channels" : 32, "stride" : 2},
                    {"kernel_size" : 4, "out_channels" : 64, "stride" : 2},
                    {"kernel_size" : 3, "out_channels" : 64, "stride" : 1},
                    
                ]
            }
        }
    ]
}


def hyperparameter_suggestions():

    from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.disjoint_hp_suggestion import DisjointHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.complex_hp_suggestion import ComplexHpSuggestion
    from automarl.components.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.list_hp_suggestion import DictHyperparameterSuggestion


    return [

        SingleHyperparameterSuggestion(
            name="times_to_learn",
            value_suggestion= ("int",  {"low" : 4, "high" : 16 }),
            hyperparameter_localizations= [
                ["input", "rl_trainer", 1, "agents_trainers_input", "times_to_learn"]            
            ]

        ),

        SingleHyperparameterSuggestion(
            name="batch_size",
            value_suggestion= ("cat",  {"choices" : [16, 32, 64]}),
            hyperparameter_localizations= [
                ["input", "rl_trainer", 1, "agents_trainers_input", "batch_size"]            
            ]

        ),

        SingleHyperparameterSuggestion(
            name="optimization_interval",
            value_suggestion= ("int",  {"low" : 128, "high" : 1024 }),
            hyperparameter_localizations= [
                ["input", "rl_trainer", 1, "agents_trainers_input", "optimization_interval"],
                ["input", "rl_trainer", 1, "agents_trainers_input", "memory", 1, "capacity"],
            ]

        ),

        SingleHyperparameterSuggestion(
            name="lr_strat",
            hyperparameter_localizations=[
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "optimizer", 1, "linear_decay_learning_rate_with_final_input_value_of"]
                ],
            value_suggestion=("cat", {"choices" : [
                                                    None,
                                                ('relative', 
                                                [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "__any__"]})]
                                                )           
                                                   ]})
            
        ),

        SingleHyperparameterSuggestion(
            name="lr_value",
            hyperparameter_localizations=[
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "optimizer", 1, "learning_rate"]
                                          ],
            value_suggestion=("float", {"low" : 1.5e-08, "high" :0.09})
        ),

        SingleHyperparameterSuggestion(
            name="clip_epsilon",
            hyperparameter_localizations=[
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "clip_epsilon"]
                ],
            value_suggestion=("float", {"low" : 0.1, "high" : 0.3})
        ),

        SingleHyperparameterSuggestion(
            name="value_loss_coef",
            hyperparameter_localizations=[
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "value_loss_coef"]
                        ],
            value_suggestion=("float", {"low" : 0.3, "high" : 0.7})
        ),

        SingleHyperparameterSuggestion(
            name="lambda_gae",
            hyperparameter_localizations=["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "lambda_gae"],
            value_suggestion=("float", {"low" : 0.7, "high" : 0.999})
        ),


        DisjointHyperparameterSuggestion(
            name = "entropy_coef_strat",
            hyperparameter_localizations= [
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "entropy_coef"]
            ],
            allow_none = False,
            disjoint_hyperparameter_suggestions= [
                SingleHyperparameterSuggestion(
                    name="value",
                    value_suggestion=("float", {"low" : 0.0, "high" : 0.3})
                ),
                ComplexHpSuggestion(
                    "dynamic_struc",
                    structure_to_add=[
                        str(DynamicLinearValueInRangeBasedOnComponent),
                        {
                            "input_for_fun_key" : "optimizations_done",
                            "initial_value" : 0.2,
                            "final_value" : 0,
                            "input_component" : ('relative', [("__get_by_name__", {"name_of_component" : "AdamOptimizerComponent"})]),
                            "input_for_fun_max_value" : 
                             ('relative', 
                              [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "agent"]})]
                             )
                        }
                    ],
                    actual_hyperparameter_suggestion= SingleHyperparameterSuggestion(
                        name="value",
                        value_suggestion=("float", {"low" : 0.01, "high" : 0.4}),
                        hyperparameter_localizations=[[1, "initial_value"]]
                    )

                )

            ]
        ),


        DisjointHyperparameterSuggestion(
            name = "clip_grad_strat",
            hyperparameter_localizations= [
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "optimizer", 1, "clip_grad_value"]
            ],
            allow_none = True,
            disjoint_hyperparameter_suggestions= [
                SingleHyperparameterSuggestion(
                    name="value",
                    value_suggestion=("float", {"low" : 0.05, "high" : 1.0})
                ),
                ComplexHpSuggestion(
                    "dynamic_struc",
                    structure_to_add=[
                        str(DynamicLinearValueInRangeBasedOnComponent),
                        {
                            "input_for_fun_key" : "optimizations_done",
                            "initial_value" : 0.2,
                            "final_value" : 0,
                            "input_component" : ('relative', [("__get_by_name__", {"name_of_component" : "AdamOptimizerComponent"})]),
                            "input_for_fun_max_value" : 
                             ('relative', 
                              [("__get_by_name__", {"name_of_component" : "RLTrainerComponent"}), ("__get_exposed_value__", {"value_localization" : ["optimizations_to_do_per_agent", "__any__"]})]
                             )
                        }
                    ],
                    actual_hyperparameter_suggestion= SingleHyperparameterSuggestion(
                        name="value",
                        value_suggestion=("float", {"low" : 0.05, "high" : 0.5}),
                        hyperparameter_localizations=[[1, "initial_value"]]
                    )

                )

            ]
        ),

        VariableListHyperparameterSuggestion(
                                            name="policy_head_layers",
                                            min_len=1,
                                            max_len=3,
                                            hyperparameter_localizations=[["input", "agents_input", "policy", 1, "model", 1, "models", 1, 1, "layers"]], # the localization of the actor head
                                            hyperparameter_suggestion_for_list=SingleHyperparameterSuggestion( value_suggestion=("cat", {"choices" : [8, 16, 32, 64, 128]}))
                                            ),

        VariableListHyperparameterSuggestion(
                                            name="critic_head_layers",
                                            min_len=1,
                                            max_len=2,
                                            hyperparameter_localizations=[["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "critic_model", 1, "models", 1, 1, "layers"]], # the localization of the actor head
                                            hyperparameter_suggestion_for_list=SingleHyperparameterSuggestion( value_suggestion=("cat", {"choices" : [8, 16, 32, 64]}))
                                            ),


        VariableListHyperparameterSuggestion(
                    name='shared_layers',
                    hyperparameter_localizations=[["child_components", 0, "input", "cnn_layers"]],
                    min_len=2,
                    max_len=4,
                    hyperparameter_suggestion_for_list=DictHyperparameterSuggestion(
                        hyperparameter_suggestions={
                            "out_channels" : SingleHyperparameterSuggestion(name="out_channels",
                                                           value_suggestion=("cat", {"choices": [16, 32, 64, 128]})
                            ),
                            "kernel_size" : SingleHyperparameterSuggestion(name="kernel_size",
                                                           value_suggestion=("cat", {"choices": [3, 4, 8]})
                            ),
                            "stride" : SingleHyperparameterSuggestion(name="stride",
                                                           value_suggestion=("cat", {"choices": [1, 2]})
                            )
                        }
                    )
                ),
                                            
        DisjointHyperparameterSuggestion(
            name = "model_init_strat",
            hyperparameter_localizations= [
                ["input", "agents_input", "policy", 1, "model", 1, "models", 1, 1, "parameters_initialization_strategy"], # the localization of the actor head
                ["input", "rl_trainer", 1, "agents_trainers_input", "learner", 1, "critic_model", 1, "models", 1, 1, "parameters_initialization_strategy"], # critic head
                ["child_components", 0, "input", "parameters_initialization_strategy"]
            ],
            allow_none = True,
            disjoint_hyperparameter_suggestions= [

                ComplexHpSuggestion(
                    "orthogonal_init",
                    structure_to_add=[
                        str(TorchModelInitializationStrategyOrthogonal),
                        {
                            "gain" : 0.1
                        }
                    ],
                    actual_hyperparameter_suggestion= SingleHyperparameterSuggestion(
                        name="gain",
                        value_suggestion=("float", {"low" : 0.05, "high" : 0.2}),
                        hyperparameter_localizations=[[1, "gain"]]
                    )

                )

            ]
        )

    ]