from automarl.components.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.joint_model import ModelSequenceComponent
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.rl.learners.ppo_learner import PPOLearner

from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer
from automarl.components.rl.trainers.agent_trainer.agent_trainer_ppo import AgentTrainerPPO

from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator

from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_acessories import AgentTrainerConvergenceDetector
from automarl.components.rl.environment.pettingzoo.aec_pettingzoo_env import AECPettingZooEnvironmentWrapper
from automarl.components.rl.policy.stochastic_policy import MaskedCategoricalStochasticPolicy
from automarl.components.rl.evaluators.rl_vs_agents_evaluator import AgentVsAgentsWithPolicy
from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.components.rl.policy.random_policy import RandomPolicyMasked
from automarl.components.rl.evaluators.rl_agent_iter_evaluator import RLAgentIterEvaluator

# TODO: SHARED OPTIMIZER AND PREDICTIONS

def config_dict():

    return {

    "__type__": RLPipelineComponent,
    "name": "RLPipelineComponent",

    "input": {

        "do_full_setup_of_seed": True,
        "device": "cuda",

        "environment": (
            AECPettingZooEnvironmentWrapper,
            {
                "environment": "connect_four"
            }
        ),

        "agents_input": {

            "state_translator": (ToTorchTranslator, {}),

            "policy": (
                MaskedCategoricalStochasticPolicy,
                {
                    "model": (
                        ModelSequenceComponent,
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

                            ],
                            "child_components" :         
                            [
                                (FullyConnectedModelSchema,
                                    {
                                        "name" : "shared_model",
                                         "layers": [256, 256]
                                    })]
                        }
                    )
                }
            )
        },

        "rl_trainer": ( RLTrainerComponent, {
            
            "name": "RLTrainerComponent",

            "limit_total_steps": 100_000,

            "predict_optimizations_to_do": True,

            "default_trainer_class": AgentTrainerPPO,

            "agents_trainers_input": {

                "optimization_interval": 512,

                "times_to_learn": 10,

                "batch_size": 128,

                "discount_factor": 0.99,

                "learn_with_all_memory" : True,

                "learner": (

                    PPOLearner,

                    {

                        "lambda_gae": 0.95,

                        "clip_epsilon": 0.2,

                        "entropy_coef": 0.01,

                        "value_loss_coef": 0.5,

                        "critic_model": (

                            ModelSequenceComponent,

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
                                        "final_value": 0.1,
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
                    }
                ),

                "memory": (

                    TorchMemoryComponent,

                    {
                        "capacity": 512
                    }

                ),

                "agent_trainer_acessories" : [
                    (AgentTrainerConvergenceDetector, {
                        "standard_deviation_treshold" : 0.5, # less than this is convergence
                        "min_avg_value" : 0.5,
                        "n_values_to_use" : 50,
                        "value_key" : "episode_reward"
                    })
                ]

            }

            }

        ),

        "evaluation_report_strategy" : "best",

        "component_evaluator" : (
            RLAgentIterEvaluator, {
                "single_agent_evaluators" : [

                (AgentVsAgentsWithPolicy,
                            {
                                "policy_type_for_others" : RandomPolicyMasked,
                                "number_of_episodes" : 200,
                                "rl_player_definition" : (RLPlayer, {}),
                                "base_evaluator" : (LastValuesAvgStdEvaluator, {"std_deviation_factor" : 100})
                            }
                )

                ]

            }
            
        )

    },

}

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
        "n_steps" : 10,
        "use_best_component_strategy_with_index" : 5,
        "do_initial_evaluation" : True,
        "pruner" : [OptunaPrunerWrapper, {"optuna_pruner" : "Percentile", "pruner_input" : {"percentile" : 90.0, "n_warmup_steps" : 5}}],
        "sampler" : [OptunaSamplerWrapper, {"optuna_sampler" : "TreeParzen", "sampler_input" : {"n_startup_trials" : 50, "multivariate": True, "group": True}}]
    }

def hyperparameters_to_optimize():
    '''
    The recomended order of hyperparameters to focus on trying to optimize, with the others using the recomended values
    '''
    
    return [
        [10, ["learn_with_all_memory", "times_to_learn", "steps_for_agent_trainer_exec", "optimization_interval", "batch_size"]],
        [10, ["learning_rate", "clip_epsilon", "value_loss_coef", "lambda_gae", "entropy_coef_strat", "clip_grad_strat"]],
        [10, ["policy_head_layers", "critic_head_layers", "shared_layers"]]
    ]



def hyperparameter_suggestions():

    '''
    The recomended hyperparameter suggestions
    '''

    from automarl.components.hp_opt.hp_suggestion.single_hp_suggestion import SingleHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.disjoint_hp_suggestion import DisjointHyperparameterSuggestion
    from automarl.components.hp_opt.hp_suggestion.complex_hp_suggestion import ComplexHpSuggestion
    from automarl.components.hp_opt.hp_suggestion.variable_list_hp_suggestion import VariableListHyperparameterSuggestion

    agents_input = ["input","agents_input"]
    agents_trainer_input = ["input","rl_trainer",1,"agents_trainers_input"]
    learner_input = [*agents_trainer_input, "learner", 1]

    policy_input = [*agents_input, "policy", 1]
    
    policy_model_input = [*policy_input, "model", 1]
    policy_head = [*policy_model_input, "models", 1]
    policy_head_input = [*policy_head, 1]

    shared_model = [*policy_model_input, "child_components", 0]
    shared_model_input = [*shared_model, 1]

    critic_model_models = [*learner_input, "critic_model", 1, "models"]
    critic_head_input = [*critic_model_models, 1, 1]


    return [

        SingleHyperparameterSuggestion(
            name="learn_with_all_memory",
            value_suggestion=("cat", {"choices": [True, False]}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "learn_with_all_memory"]
            ]
        ),

        # PPO EPOCHS
        SingleHyperparameterSuggestion(
            name="times_to_learn",
            value_suggestion=("int", {"low": 2, "high": 20}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "times_to_learn"]
            ]
        ),

        # PPO ROLLOUT SIZE
        SingleHyperparameterSuggestion(
            name="optimization_interval",
            value_suggestion=("int", {"low": 256, "high": 1024, "step" : 64}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","optimization_interval"],
                ["input","rl_trainer",1,"agents_trainers_input","memory",1,"capacity"],
            ]
        ),

        # PPO MINIBATCH SIZE
        SingleHyperparameterSuggestion(
            name="batch_size",
            value_suggestion=("cat", {"choices": [64,128]}),
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
            max_len=2,
            hyperparameter_localizations=[
                [
                    *policy_head_input, "layers"
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
                    *critic_head_input, "layers"
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
            max_len=2,
            hyperparameter_localizations=[
                [
                    *shared_model_input, "layers"
                ]
            ],
            hyperparameter_suggestion_for_list=
                SingleHyperparameterSuggestion(
                    value_suggestion=("cat",{"choices":[64,128,256,512]})
                )
        )

    ]