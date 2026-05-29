from automarl.components.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.joint_model import ModelSequenceComponent
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema

from automarl.components.rl.policy.stochastic_policy import ConstrainedNormalStochasticPolicy, NormalStochasticPolicy
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer
from automarl.components.rl.trainers.agent_trainer.agent_trainer_ppo import AgentTrainerPPOCriticAware
from automarl.components.rl.environment.pettingzoo.parallel_petting_zoo_env import PettingZooEnvironmentWrapperParallel

from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.rl.evaluators.rl_evaluator_player import EvaluatorWithPlayer
from automarl.components.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator

from automarl.components.rl.rl_player.rl_parallel_player import RLParallelPlayer
from automarl.components.rl.trainers.rl_trainer.rl_trainer_mappo import RLTrainerMAPPO
from automarl.components.rl.learners.ppo_learner_separated import PPOLearnerNoCritic, PPOLearnerOnlyCritic


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
                    "model": (
                        ModelSequenceComponent,
                        {
                            "name": "policy_model",
                            "models": [

                                    ('absolute',[
                                        ("__get_by_name__", {"name_of_component": "shared_model"})
                                    ]),

                                (FullyConnectedModelSchema,
                                {
                                    "name": "policy_head",
                                    "layers": [256, 256, 64]
                                })

                            ]
                        }
                    )
                }
            )
        },

        "rl_trainer": (

            RLTrainerMAPPO,

            {

                "name": "RLTrainerComponent",

                "limit_total_steps": 1_000_000,

                "predict_optimizations_to_do": True,

                "default_trainer_class": AgentTrainerPPOCriticAware,

                "discount_factor": 0.99,

                "optimization_interval": 2048,

                "times_to_learn": 10,

                "batch_size": 128,

                "learn_with_all_memory" : True,

                "critic_learner" : (PPOLearnerOnlyCritic, {

                    "clip_epsilon" : 0.2,

                    "value_loss_coef" : 0.5,

                    "lambda_gae" : 0.95,

                    "critic_model" : (
                    FullyConnectedModelSchema,
                    {"layers": [256, 128, 64]}
                        ),

                    "optimizer" : (
                        AdamOptimizer,
                        {           "name" : "CriticOptimizer",
                         
                                "learning_rate": 3e-4,

                                "linear_decay_learning_rate_with_final_input_value_of":
                                ('relative',
                                    [
                                        ("__get_by_type__", {"type": PPOLearnerOnlyCritic}),
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
                                            "input_component": ('relative', ("__get_by_name__", {"name_of_component": "CriticOptimizer"})),

                                            "input_for_fun_max_value":
                                            ('relative',
                                                [
                                                    ("__get_by_type__", {"type": PPOLearnerOnlyCritic}),
                                                    ("__get_exposed_value__", {
                                                        "value_localization": ["optimizations_to_do"]
                                                    })
                                        ]
                                            )
                                        }
                                    )

                        }
                    ),


                }),

            "agents_trainers_input": {
        
                "times_to_learn": 10,

                "batch_size": 128,

                "learn_with_all_memory" : True,

                "optimization_interval": 2048,

                "learner": (

                    PPOLearnerNoCritic,

                    {

                        "clip_epsilon": 0.2,

                        "entropy_coef": 0.01,

                        "lambda_gae" : 0.95,

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
                                        "input_component": ('relative', ("__get_by_name__", {"name_of_component": "AdamOptimizerComponent"})),

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

                        )
                    }
                ),

                "memory": (

                    TorchMemoryComponent,

                    {
                        "capacity": 2048
                    }

                )

            },

            "memory": (

                TorchMemoryComponent,

                {
                        "capacity": 2048
                }

            )

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
                                    "name": "shared_model",
                                    "layers": []
                                })

        ]

    }

}

def hyperparameter_optimization_definition():
    from automarl.components.hp_opt.hp_opt_strategies.hp_optimization_hyperband import HyperparameterOptimizationPipelineHyperband


    return (
        HyperparameterOptimizationPipelineHyperband,
        hyperparameter_optimization_input()
    )

def hyperparameter_optimization_input():

    '''
    The recomended input for an hyperparameter optimization
    '''

    from automarl.components.hp_opt.pruners.pruner import OptunaPrunerWrapper
    from automarl.components.hp_opt.samplers.sampler import OptunaSamplerWrapper

    return {

        "hyperparameters_range_list" : hyperparameter_suggestions(),
        "hyperparameters_to_optimize" : hyperparameters_to_optimize(),
        "n_trials" : 200,
        "n_steps" : 40,
        "eta" : 2,
        "use_best_component_strategy_with_index" : 5,
        "do_initial_evaluation" : True,
        "pruner" : [OptunaPrunerWrapper, {"optuna_pruner" : "Percentile", "pruner_input" : {"percentile" : 90.0, "n_warmup_steps" : 5}}],
        "sampler" : [OptunaSamplerWrapper, {"optuna_sampler" : "TreeParzen", "sampler_input" : {"n_startup_trials" : 80, "multivariate": True, "group": True}}]
    }

def hyperparameters_to_optimize():
    '''
    The recomended order of hyperparameters to focus on trying to optimize, with the others using the recomended values
    '''
    
    return [
        [15, ["optimization_interval", "batch_size", "times_to_learn", "times_to_learn_critic"]],
        [15, ["learning_rate", "critic_learning_rate", "clip_epsilon", "clip_epsilon_critic", "lambda_gae", "critic_lambda_gae", "value_loss_coef"]],
        [15, ["entropy_coef_strat", "clip_grad_strat", "clip_grad_strat_critic"]],
        [15, ["policy_head_layers", "critic_head_layers", "shared_layers", "state_memory_size"]],
    ]


def hyperparameter_suggestions():

    from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.disjoint_hp_suggestion import DisjointHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.complex_hp_suggestion import ComplexHpSuggestion
    from automarl.components.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion


    agents_input = ["input", "agents_input"]
    
    agents_policy = [*agents_input, "policy"]
    agents_policy_head = [*agents_policy, 1, "model", 1, "models", 1]
    agents_policy_head_input = [*agents_policy_head, 1]

    rl_trainer = ["input", "rl_trainer"]
    rl_trainer_input = [*rl_trainer, 1]
    
    rl_trainer_memory_input = [*rl_trainer_input, "memory", 1]

    rl_trainer_learner_input = [*rl_trainer_input, "critic_learner", 1]

    critic_optimizer_input = [*rl_trainer_learner_input, "optimizer", 1]

    critic_model_input = [*rl_trainer_learner_input, "critic_model", 1]

    agents_trainers_input = [*rl_trainer_input, "agents_trainers_input"]
    
    agents_learner = [*agents_trainers_input, "learner"]
    agents_learner_input = [*agents_learner, 1]

    agents_optimizer = [*agents_learner_input, "optimizer"]
    agents_optimizer_input = [*agents_optimizer, 1]

    agents_memory_input = [*agents_trainers_input, "memory", 1]

    shared_model = ["input", "child_components", 0]
    shared_model_input = [*shared_model, 1]

    return [

        SingleHyperparameterSuggestion(
            name="state_memory_size",
            value_suggestion=("int", {"low": 1, "high": 4}),
            hyperparameter_localizations=[
                [*agents_input, "state_memory_size"]
            ]
        ),

        # PPO EPOCHS
        SingleHyperparameterSuggestion(
            name="times_to_learn_critic",
            value_suggestion=("int", {"low": 4, "high": 20}),
            hyperparameter_localizations=[
                [*rl_trainer_input,"times_to_learn"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="times_to_learn",
            value_suggestion=("int", {"low": 4, "high": 20}),
            hyperparameter_localizations=[
                [*agents_trainers_input,"times_to_learn"]
            ]
        ),

        # PPO ROLLOUT SIZE
        SingleHyperparameterSuggestion(
            name="optimization_interval",
            value_suggestion=("int", {"low": 512, "high": 4096}),
            hyperparameter_localizations=[
                [*rl_trainer_input, "optimization_interval"],
                [*agents_memory_input, "capacity"],
                [*rl_trainer_memory_input, "capacity"],
                [*agents_trainers_input, "optimization_interval"],
            ]
        ),

        # PPO MINIBATCH SIZE
        SingleHyperparameterSuggestion(
            name="batch_size",
            value_suggestion=("cat", {"choices": [64,128,256]}),
            hyperparameter_localizations=[
                [*agents_trainers_input,"batch_size"],
                [*rl_trainer_input, "batch_size"]
            ]
        ),

        # LEARNING RATE
        SingleHyperparameterSuggestion(
            name="learning_rate",
            value_suggestion=("float", {"low":1e-5,"high":3e-3,"log":True}),
            hyperparameter_localizations=[
                [*agents_optimizer_input,"learning_rate"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="critic_learning_rate",
            value_suggestion=("float", {"low":1e-5,"high":3e-3,"log":True}),
            hyperparameter_localizations=[
                [*critic_optimizer_input, "learning_rate"]
            ]
        ),

        # PPO CLIP
        SingleHyperparameterSuggestion(
            name="clip_epsilon_critic",
            value_suggestion=("float", {"low":0.1,"high":0.3}),
            hyperparameter_localizations=[
                [*rl_trainer_learner_input,"clip_epsilon"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="clip_epsilon",
            value_suggestion=("float", {"low":0.1,"high":0.3}),
            hyperparameter_localizations=[
                [*agents_learner_input,"clip_epsilon"]
            ]
        ),

        # VALUE LOSS WEIGHT
        SingleHyperparameterSuggestion(
            name="value_loss_coef",
            value_suggestion=("float", {"low":0.3,"high":0.7}),
            hyperparameter_localizations=[
                [*rl_trainer_learner_input,"value_loss_coef"]
            ]
        ),

        # GAE
        SingleHyperparameterSuggestion(
            name="critic_lambda_gae",
            value_suggestion=("float", {"low":0.9,"high":0.99, "log" : True}),
            hyperparameter_localizations=[
                [*rl_trainer_learner_input,"lambda_gae"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="lambda_gae",
            value_suggestion=("float", {"low":0.9,"high":0.99, "log" : True}),
            hyperparameter_localizations=[
                [*agents_learner_input,"lambda_gae"]
            ]
        ),

        # ENTROPY COEFFICIENT
        DisjointHyperparameterSuggestion(
            name="entropy_coef_strat",
            hyperparameter_localizations=[
                [*agents_learner_input,"entropy_coef"]
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
                            "input_for_fun_key":"optimizations_done",
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
                [*agents_optimizer_input,"clip_grad_value"],
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
                                        ("__get_by_type__", {"type": PPOLearnerOnlyCritic}),
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

        DisjointHyperparameterSuggestion(
            name="clip_grad_strat_critic",
            hyperparameter_localizations=[
                [*critic_optimizer_input,"clip_grad_value"],
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
                                ("__get_by_name__",{"name_of_component":"CriticOptimizer"})
                            ]),
                            "input_for_fun_max_value":
                            ('relative',[
                                        ("__get_by_type__", {"type": PPOLearnerOnlyCritic}),
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
            max_len=4,
            hyperparameter_localizations=[
                [*agents_policy_head_input,"layers"]
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
            max_len=3,
            hyperparameter_localizations=[
                [*critic_model_input,"layers"]
            ],
            hyperparameter_suggestion_for_list=
                SingleHyperparameterSuggestion(
                    value_suggestion=("cat",{"choices":[32,64,128,256]})
                )
        ),

        # SHARED NETWORK
        VariableListHyperparameterSuggestion(
            name="shared_layers",
            min_len=0,
            max_len=1,
            hyperparameter_localizations=[
                [*shared_model_input,"layers"]
            ],
            hyperparameter_suggestion_for_list=
                SingleHyperparameterSuggestion(
                    value_suggestion=("cat",{"choices":[128,256,512]})
                )
        ),

    ]