from automarl.components.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent

from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer

from automarl.components.rl.policy.stochastic_policy import CategoricalStochasticPolicy

from automarl.components.rl.rl_pipeline import RLPipelineComponent

from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer
from automarl.components.rl.trainers.agent_trainer.agent_trainer_ppo import AgentTrainerPPO

from automarl.components.rl.learners.ppo_learner import PPOLearner

from automarl.components.rl.environment.gymnasium.aec_gymnasium_env import AECGymnasiumEnvironmentWrapper
from automarl.components.rl.trainers.rl_trainer.single_rl_trainer import SingleRLTrainer
from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.rl.evaluators.rl_evaluator_player import EvaluatorWithPlayer
from automarl.components.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator
from automarl.components.rl.rl_player.rl_single_player import RLSinglePlayer

def experiment_name():
    return "montaincar_ppo"

def config_dict():

    return {

        "__type__": RLPipelineComponent,
        "name": "RLPipelineComponent",

        "input": {

            "device": "cuda",

            "stop_at_evaluation" : 499.99,

            "environment": (
                AECGymnasiumEnvironmentWrapper,
                {
                    "environment": "MountainCar-v0"
                }
            ),

            "agents_input": {

                "policy": (
                    CategoricalStochasticPolicy,
                    {
                        "model": (
                            FullyConnectedModelSchema,
                            {
                                "layers": [64,64]
                            }
                        )
                    }
                ),

            "state_translator" : (ToTorchTranslator, {})

            },

            "rl_trainer": (

                SingleRLTrainer,

                {
                    "name": "RLTrainerComponent",

                    "limit_total_steps": 1_000_000,

                    "predict_optimizations_to_do": True,

                    "default_trainer_class": AgentTrainerPPO,

                    "agents_trainers_input": {

                        "discount_factor": 0.99,

                        "optimization_interval": 2048,

                        "times_to_learn": 4,

                        "batch_size": 128,

                        "learn_with_all_memory": True,

                        "learner": (

                            PPOLearner,

                            {

                                "clip_epsilon": 0.2,

                                "entropy_coef": 0.0,

                                "lambda_gae": 0.98,

                                "value_loss_coef": 0.5,

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

                                        "clip_grad_value": 0.5

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

                    }

                }

            ),

            "component_evaluator" : (
            EvaluatorWithPlayer,
            {
                "number_of_episodes" : 5,
                "rl_player_definition" : (RLSinglePlayer, {}),
                "base_evaluator" : (LastValuesAvgStdEvaluator, {"value_to_use" : "episode_reward"})
            }
        )


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
        "steps" : 10,
        "hyperband_eta" : 2,
        "use_best_component_strategy_with_index" : 5,
        "do_initial_evaluation" : True,
        "pruner" : [OptunaPrunerWrapper, {"optuna_pruner" : "Percentile", "pruner_input" : {"percentile" : 90.0, "n_warmup_steps" : 5}}],
        "sampler" : [OptunaSamplerWrapper, {"optuna_sampler" : "TreeParzen", "sampler_input" : {"n_startup_trials" : 80, "multivariate": True, "group": True}}]
    }


def hyperparameters_to_optimize():

    return [

        [15, [
            "optimization_interval",
            "batch_size",
            "times_to_learn"
        ]],

        [15, [
            "learning_rate",
            "clip_epsilon",
            "lambda_gae",
            "discount_factor",
            "value_loss_coef"
        ]],

        [15, [
            "entropy_coef_strat",
            "clip_grad_strat"
        ]],

        [15, [
            "network_layers"
        ]]

    ]



def hyperparameter_suggestions():

    from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.disjoint_hp_suggestion import DisjointHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.complex_hp_suggestion import ComplexHpSuggestion
    from automarl.components.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion

    from automarl.components.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent

    from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer

    agents_input = ["input", "agents_input"]

    policy = [*agents_input, "policy"]
    policy_model_input = [*policy, 1, "model", 1]

    rl_trainer = ["input", "rl_trainer"]
    rl_trainer_input = [*rl_trainer, 1]

    agents_trainers_input = [*rl_trainer_input, "agents_trainers_input"]

    learner = [*agents_trainers_input, "learner"]
    learner_input = [*learner, 1]

    optimizer = [*learner_input, "optimizer"]
    optimizer_input = [*optimizer, 1]

    memory_input = [*agents_trainers_input, "memory", 1]

    return [

        # GAMMA

        SingleHyperparameterSuggestion(
            name="discount_factor",
            value_suggestion=(
                "float",
                {
                    "low": 0.97,
                    "high": 0.999,
                    "log": True
                }
            ),
            hyperparameter_localizations=[
                [*agents_trainers_input, "discount_factor"]
            ]
        ),

        # PPO ROLLOUT SIZE

        SingleHyperparameterSuggestion(
            name="optimization_interval",
            value_suggestion=(
                "cat",
                {
                    "choices": [1024, 2048, 4096]
                }
            ),
            hyperparameter_localizations=[
                [*agents_trainers_input, "optimization_interval"],
                [*memory_input, "capacity"]
            ]
        ),

        # PPO EPOCHS

        SingleHyperparameterSuggestion(
            name="times_to_learn",
            value_suggestion=(
                "int",
                {
                    "low": 4,
                    "high": 20
                }
            ),
            hyperparameter_localizations=[
                [*agents_trainers_input, "times_to_learn"]
            ]
        ),

        # MINIBATCH SIZE

        SingleHyperparameterSuggestion(
            name="batch_size",
            value_suggestion=(
                "cat",
                {
                    "choices": [64, 128, 256]
                }
            ),
            hyperparameter_localizations=[
                [*agents_trainers_input, "batch_size"]
            ]
        ),

        # LEARNING RATE

        SingleHyperparameterSuggestion(
            name="learning_rate",
            value_suggestion=(
                "float",
                {
                    "low": 1e-5,
                    "high": 3e-3,
                    "log": True
                }
            ),
            hyperparameter_localizations=[
                [*optimizer_input, "learning_rate"]
            ]
        ),

        # PPO CLIP

        SingleHyperparameterSuggestion(
            name="clip_epsilon",
            value_suggestion=(
                "float",
                {
                    "low": 0.1,
                    "high": 0.3
                }
            ),
            hyperparameter_localizations=[
                [*learner_input, "clip_epsilon"]
            ]
        ),

        # GAE

        SingleHyperparameterSuggestion(
            name="lambda_gae",
            value_suggestion=(
                "float",
                {
                    "low": 0.90,
                    "high": 0.99,
                    "log": True
                }
            ),
            hyperparameter_localizations=[
                [*learner_input, "lambda_gae"]
            ]
        ),

        # VALUE LOSS COEF

        SingleHyperparameterSuggestion(
            name="value_loss_coef",
            value_suggestion=(
                "float",
                {
                    "low": 0.3,
                    "high": 1.0
                }
            ),
            hyperparameter_localizations=[
                [*learner_input, "value_loss_coef"]
            ]
        ),

        # ENTROPY

        DisjointHyperparameterSuggestion(
            name="entropy_coef_strat",
            hyperparameter_localizations=[
                [*learner_input, "entropy_coef"]
            ],
            allow_none=False,
            disjoint_hyperparameter_suggestions=[

                SingleHyperparameterSuggestion(
                    name="value",
                    value_suggestion=(
                        "float",
                        {
                            "low": 0.0,
                            "high": 0.02
                        }
                    )
                ),

                ComplexHpSuggestion(
                    "dynamic_struc",
                    structure_to_add=[
                        str(DynamicLinearValueInRangeBasedOnComponent),
                        {
                            "input_for_fun_key": "optimizations_done",
                            "initial_value": 0.01,
                            "final_value": 0.0,
                            "input_component": (
                                'relative',
                                [
                                    (
                                        "__get_by_name__",
                                        {
                                            "name_of_component": "AdamOptimizerComponent"
                                        }
                                    )
                                ]
                            ),
                            "input_for_fun_max_value":
                            (
                                'relative',
                                [
                                    (
                                        "__get_by_type__",
                                        {
                                            "type": AgentTrainer
                                        }
                                    ),
                                    (
                                        "__get_exposed_value__",
                                        {
                                            "value_localization": [
                                                "optimizations_to_do"
                                            ]
                                        }
                                    )
                                ]
                            )
                        }
                    ],
                    actual_hyperparameter_suggestion=
                    SingleHyperparameterSuggestion(
                        name="value",
                        value_suggestion=(
                            "float",
                            {
                                "low": 0.005,
                                "high": 0.02
                            }
                        ),
                        hyperparameter_localizations=[
                            [1, "initial_value"]
                        ]
                    )
                )

            ]
        ),

        # GRADIENT CLIPPING

        DisjointHyperparameterSuggestion(
            name="clip_grad_strat",
            hyperparameter_localizations=[
                [*optimizer_input, "clip_grad_value"]
            ],
            allow_none=True,
            disjoint_hyperparameter_suggestions=[

                SingleHyperparameterSuggestion(
                    name="value",
                    value_suggestion=(
                        "float",
                        {
                            "low": 0.1,
                            "high": 1.0
                        }
                    )
                ),

                ComplexHpSuggestion(
                    "dynamic_struc",
                    structure_to_add=[
                        str(DynamicLinearValueInRangeBasedOnComponent),
                        {
                            "input_for_fun_key": "optimizations_done",
                            "initial_value": 0.5,
                            "final_value": 0.1,
                            "input_component": (
                                'relative',
                                [
                                    (
                                        "__get_by_name__",
                                        {
                                            "name_of_component":
                                            "AdamOptimizerComponent"
                                        }
                                    )
                                ]
                            ),
                            "input_for_fun_max_value":
                            (
                                'relative',
                                [
                                    (
                                        "__get_by_type__",
                                        {
                                            "type": AgentTrainer
                                        }
                                    ),
                                    (
                                        "__get_exposed_value__",
                                        {
                                            "value_localization": [
                                                "optimizations_to_do"
                                            ]
                                        }
                                    )
                                ]
                            )
                        }
                    ],
                    actual_hyperparameter_suggestion=
                    SingleHyperparameterSuggestion(
                        name="value",
                        value_suggestion=(
                            "float",
                            {
                                "low": 0.2,
                                "high": 0.8
                            }
                        ),
                        hyperparameter_localizations=[
                            [1, "initial_value"]
                        ]
                    )
                )

            ]
        ),

        # NETWORK ARCHITECTURE

        VariableListHyperparameterSuggestion(
            name="network_layers",
            min_len=1,
            max_len=3,
            hyperparameter_localizations=[
                [*policy_model_input, "layers"]
            ],
            hyperparameter_suggestion_for_list=
            SingleHyperparameterSuggestion(
                value_suggestion=(
                    "cat",
                    {
                        "choices": [64, 128, 256]
                    }
                )
            )
        )

    ]
