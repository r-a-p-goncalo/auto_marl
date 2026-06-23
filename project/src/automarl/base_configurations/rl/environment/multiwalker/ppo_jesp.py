from automarl.components.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.joint_model import ModelSequenceComponent
from automarl.components.ml.models.model_initialization_strategy import TorchModelInitializationStrategyOrthogonal
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.rl.learners.ppo_learner import PPOLearner

from automarl.components.rl.policy.stochastic_policy import ConstrainedNormalStochasticPolicy
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer
from automarl.components.rl.trainers.agent_trainer.agent_trainer_ppo import AgentTrainerPPO
from automarl.components.rl.environment.pettingzoo.parallel_petting_zoo_env import PettingZooEnvironmentWrapperParallel

from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.rl.evaluators.rl_evaluator_player import EvaluatorWithPlayer
from automarl.components.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator

from automarl.components.rl.rl_player.rl_parallel_player import RLParallelPlayer
from automarl.components.rl.trainers.rl_trainer.jesp_rl_trainer import JESPParalelTrainer
from automarl.components.rl.learners.convergence_detectors.avg_out_convergence_detector import ConvergenceDetector
from automarl.components.rl.trainers.agent_trainer.agent_trainer_acessories import AgentTrainerConvergenceDetector, AgentTrainerSlopeConvergenceDetector

# TODO: SHARED OPTIMIZER AND PREDICTIONS

def experiment_name():
    return "multiwalker_ppo_jesp"

def config_dict():

    return {

    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",

    "input": {

        "do_full_setup_of_seed": True,
        "device": "cuda",

        "environment": (
            PettingZooEnvironmentWrapperParallel,
            {
                "environment": "multiwalker"
            }
        ),

        "agents_input": {

            "state_memory_size" : 2,

            "state_translator": (ToTorchTranslator, {}),

            "policy": (
                ConstrainedNormalStochasticPolicy,
                {

                    "model": ('absolute',[
                                        ("__get_by_name__", {"name_of_component": "policy_model"})
                                        ]
                                        )
                }
            )
        },

        "rl_trainer": ( JESPParalelTrainer, {
            
            "name": "RLTrainerComponent",

            "limit_total_steps": 10_000_000,

            "predict_optimizations_to_do": True,

            "default_trainer_class": AgentTrainerPPO,

            "agents_trainers_input": {

                "optimization_interval": 2048,

                "times_to_learn": 10,

                "batch_size": 128,

                "discount_factor": 0.99,

                "limit_steps" : 25_000,

                "learn_with_all_memory" : True,

                "learner": (

                    PPOLearner,

                    {

                        "lambda_gae": 0.95,

                        "clip_epsilon": 0.2,

                        "entropy_coef": 0.01,

                        "value_loss_coef": 0.5,

                        "critic_model": ('absolute',[
                                                    ("__get_by_name__", {"name_of_component": "critic_model"})
                                                    ]
                                        ),

                        "optimizer": (

                            AdamOptimizer,

                            {

                                "name": "AdamOptimizerComponent",

                                "learning_rate": 3e-4,

                                "linear_decay_learning_rate_with_final_input_value_of":
                                ('relative',
                                    [
                                        ("__get_by_type__", {"type": AgentTrainer}),
                                        ("__get_exposed_value__", {
                                            "value_localization": ["optimizations_to_do"]
                                        })
                                    ]
                                ),

                                "clip_grad_value":

                                (
                                    DynamicLinearValueInRangeBasedOnComponent,
                                    {
                                        "input_for_fun_key": "optimizations_done",
                                        "initial_value": 0.5,
                                        "final_value": 0,
                                        "input_component": ('relative', [("__get_by_name__", {"name_of_component": "AdamOptimizerComponent"})]),

                                        "input_for_fun_max_value":
                                        ('relative',
                                            [
                                                ("__get_by_type__", {"type": AgentTrainer}),
                                                ("__get_exposed_value__", {
                                                    "value_localization": ["optimizations_to_do"]
                                                })
                                    ]
                                        )
                                    }
                                )
                            }

                        ),

                    "learning_acessories" : [
                        ( ConvergenceDetector,
                            {
                                "memory_size" : 50,
                                "convergence_treshold" : 0.05, # if average difference between logs is less than this, is convergence, evaluated at learning time
                                "old_values_new_values_keys" : ["log_prob", "new_log_probs"]
                            }
                        )
                    ]
                    }
                ),

                "memory": (

                    TorchMemoryComponent,

                    {
                        "capacity": 2048
                    }

                ),

                "agent_trainer_acessories" : [
                    (AgentTrainerConvergenceDetector, {
                        "standard_deviation_treshold" : 15, # less than this is convergence
                        "n_values_to_use" : 200
                    }),
                    (AgentTrainerSlopeConvergenceDetector, {
                        "slope_threshold" : 0.1, # less than this is convergence
                        "n_values_to_use" : 200
                    })
                ]

            }

            }

        ),

        "component_evaluator" : (
            EvaluatorWithPlayer,
            {
                "number_of_episodes" : 5,
                "rl_player_definition" : (RLParallelPlayer, {}),
                "base_evaluator" : (LastValuesAvgStdEvaluator, {"value_to_use" : "episode_reward"})
            }
        ),

        "child_components" : [

                    (FullyConnectedModelSchema,
                    {
                        "name" : "shared_model",
                                 "layers": [256, 256]
                    }),

                    (  ModelSequenceComponent,
                        {
                            "name": "policy_model",
                            "models": [

                                    ('relative',[
                                        ("__get_by_name__", {"name_of_component": "shared_model"})
                                    ]),

                                (FullyConnectedModelSchema,
                                {
                                    "name": "policy_head",
                                    "layers": [128, 64]
                                })

                            ]                                
                        }
                    ),

                        ( ModelSequenceComponent,
                            {
                                "name": "critic_model",
                                "models": [

                                    ('relative',[
                                        ("__get_by_name__", {"name_of_component": "shared_model"})
                                    ]),

                                    (FullyConnectedModelSchema,
                                    {
                                        "name": "critic_head",
                                        "layers": [128, 64]
                                    })

                                ]

                            }
                        )

        ]

    },

}



def hyperparameter_suggestions():

    from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.disjoint_hp_suggestion import DisjointHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.complex_hp_suggestion import ComplexHpSuggestion
    from automarl.components.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion

    shared_model_location = ["input", "child_components", 0]
    
    policy_model_location = ["input", "child_components", 1]
    policy_model_head_location = [*policy_model_location, 1, "models", 1]

    critic_model_location = ["input", "child_components", 2]
    critic_model_head_location = [*critic_model_location, 1, "models", 1]

    agents_trainer_input = ["input","rl_trainer",1,"agents_trainers_input"]

    return [

        SingleHyperparameterSuggestion(
            name="learn_with_all_memory",
            value_suggestion=("cat", {"choices": [True, False]}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "learn_with_all_memory"]
            ]
        ),


        SingleHyperparameterSuggestion(
            name="state_memory_size",
            value_suggestion=("int", {"low": 1, "high": 4}),
            hyperparameter_localizations=[
                ["input","agents_input","state_memory_size"]
            ]
        ),

        # PPO EPOCHS
        SingleHyperparameterSuggestion(
            name="times_to_learn",
            value_suggestion=("int", {"low": 4, "high": 20}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "times_to_learn"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="episodes_for_agent_to_learn",
            value_suggestion=("int", {"low": 15_000, "high": 50_000}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "limit_steps"]
            ]
        ),

        # PPO ROLLOUT SIZE
        SingleHyperparameterSuggestion(
            name="optimization_interval",
            value_suggestion=("int", {"low": 512, "high": 4096}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","optimization_interval"],
                ["input","rl_trainer",1,"agents_trainers_input","memory",1,"capacity"],
            ]
        ),

        # PPO MINIBATCH SIZE
        SingleHyperparameterSuggestion(
            name="batch_size",
            value_suggestion=("cat", {"choices": [64,128,256]}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","batch_size"]
            ]
        ),

        # LEARNING RATE
        SingleHyperparameterSuggestion(
            name="learning_rate",
            value_suggestion=("float", {"low":1e-5,"high":3e-3,"log":True}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","learner",1,"optimizer",1,"learning_rate"]
            ]
        ),

        # PPO CLIP
        SingleHyperparameterSuggestion(
            name="clip_epsilon",
            value_suggestion=("float", {"low":0.1,"high":0.3}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","learner",1,"clip_epsilon"]
            ]
        ),

        # VALUE LOSS WEIGHT
        SingleHyperparameterSuggestion(
            name="value_loss_coef",
            value_suggestion=("float", {"low":0.3,"high":0.7}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","learner",1,"value_loss_coef"]
            ]
        ),

        # GAE
        SingleHyperparameterSuggestion(
            name="lambda_gae",
            value_suggestion=("float", {"low":0.9,"high":0.98}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","learner",1,"lambda_gae"]
            ]
        ),

        # ENTROPY COEFFICIENT
        DisjointHyperparameterSuggestion(
            name="entropy_coef_strat",
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","learner",1,"entropy_coef"]
            ],
            allow_none=False,
            disjoint_hyperparameter_suggestions=[

                SingleHyperparameterSuggestion(
                    name="value",
                    value_suggestion=("float", {"low":0.0,"high":0.05})
                ),

                ComplexHpSuggestion(
                    "dynamic_struc",
                    structure_to_add=[
                        str(DynamicLinearValueInRangeBasedOnComponent),
                        {
                            "input_for_fun_key": "optimizations_done",
                            "initial_value":0.02,
                            "final_value":0,
                            "input_component":('relative',[
                                ("__get_by_name__",{"name_of_component":"AdamOptimizerComponent"})
                            ]),
                            "input_for_fun_max_value":
                            ('relative',[
                                        ("__get_by_type__", {"type": AgentTrainer}),
                                        ("__get_exposed_value__", {
                                            "value_localization": ["optimizations_to_do"]
                                        })
                                    ])
                        }
                    ],
                    actual_hyperparameter_suggestion=SingleHyperparameterSuggestion(
                        name="value",
                        value_suggestion=("float",{"low":0.01,"high":0.05}),
                        hyperparameter_localizations=[[1,"initial_value"]]
                    )
                )
            ]
        ),

        # GRADIENT CLIPPING
        DisjointHyperparameterSuggestion(
            name="clip_grad_strat",
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","learner",1,"optimizer",1,"clip_grad_value"]
            ],
            allow_none=True,
            disjoint_hyperparameter_suggestions=[

                SingleHyperparameterSuggestion(
                    name="value",
                    value_suggestion=("float",{"low":0.1,"high":1.0})
                ),

                ComplexHpSuggestion(
                    "dynamic_struc",
                    structure_to_add=[
                        str(DynamicLinearValueInRangeBasedOnComponent),
                        {
                            "input_for_fun_key":"optimizations_done",
                            "initial_value":0.5,
                            "final_value":0.1,
                            "input_component":('relative',[
                                ("__get_by_name__",{"name_of_component":"AdamOptimizerComponent"})
                            ]),
                            "input_for_fun_max_value":
                            ('relative',[
                                        ("__get_by_type__", {"type": AgentTrainer}),
                                        ("__get_exposed_value__", {
                                            "value_localization": ["optimizations_to_do"]
                                        })
                                    ])
                        }
                    ],
                    actual_hyperparameter_suggestion=SingleHyperparameterSuggestion(
                        name="value",
                        value_suggestion=("float",{"low":0.2,"high":0.8}),
                        hyperparameter_localizations=[[1,"initial_value"]]
                    )
                )
            ]
        ),

        # ACTOR HEAD
        VariableListHyperparameterSuggestion(
            name="policy_head_layers",
            min_len=1,
            max_len=3,
            hyperparameter_localizations=[
                [
                    *policy_model_head_location, 1, "layers"
                ]
            ],
            hyperparameter_suggestion_for_list=
                SingleHyperparameterSuggestion(
                    value_suggestion=("cat",{"choices":[32,64,128,256]})
                )
        ),

        # CRITIC HEAD
        VariableListHyperparameterSuggestion(
            name="critic_head_layers",
            min_len=1,
            max_len=2,
            hyperparameter_localizations=[
                [
                    *critic_model_head_location, 1, "layers"
                ]
            ],
            hyperparameter_suggestion_for_list=
                SingleHyperparameterSuggestion(
                    value_suggestion=("cat",{"choices":[32,64,128,256]})
                )
        ),

        # SHARED NETWORK
        VariableListHyperparameterSuggestion(
            name="shared_layers",
            min_len=1,
            max_len=4,
            hyperparameter_localizations=[
                [
                    *shared_model_location, 1, "layers"
                ]
            ],
            hyperparameter_suggestion_for_list=
                SingleHyperparameterSuggestion(
                    value_suggestion=("cat",{"choices":[64,128,256,512]})
                )
        ),

        # MODEL INITIALIZATION
        DisjointHyperparameterSuggestion(
            name="model_init_strat",
            hyperparameter_localizations=[

                [
                    *shared_model_location, 1, "parameters_initialization_strategy"
                ],
                [
                    *policy_model_head_location, 1, "parameters_initialization_strategy"
                ],
                [
                    *critic_model_head_location, 1, "parameters_initialization_strategy"
                ],
            ],
            allow_none=True,
            disjoint_hyperparameter_suggestions=[

                ComplexHpSuggestion(
                    "orthogonal_init",
                    structure_to_add=[
                        str(TorchModelInitializationStrategyOrthogonal),
                        {"gain":0.1}
                    ],
                    actual_hyperparameter_suggestion=
                        SingleHyperparameterSuggestion(
                            name="gain",
                            value_suggestion=("float",{"low":0.05,"high":0.2}),
                            hyperparameter_localizations=[[1,"gain"]]
                        )
                )
            ]
        )

    ]