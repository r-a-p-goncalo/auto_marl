from automarl.components.basic_components.dynamic_value import DynamicLinearValueInRangeBasedOnComponent
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema

from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer

from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator

from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent
from automarl.components.rl.environment.pettingzoo.aec_pettingzoo_env import AECPettingZooEnvironmentWrapper
from automarl.components.rl.evaluators.rl_vs_agents_evaluator import AgentVsAgentsWithPolicy
from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.components.rl.policy.random_policy import RandomPolicyMasked
from automarl.components.rl.evaluators.rl_agent_iter_evaluator import RLAgentIterEvaluator
from automarl.components.rl.policy.qpolicy import MaskedQPolicy
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component_dqn import AgentTrainerDQN
from automarl.components.rl.learners.q_learner import DeepQLearnerSchema, DoubleDeepQLearnerSchema
from automarl.components.rl.exploration.epsilong_greedy import EpsilonGreedyStepDecayStrategy

def experiment_name():
    return "connectorfour_dqn"

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
                MaskedQPolicy,
                {
                    "model": 
                                (FullyConnectedModelSchema,
                                {
                                    "name": "policy_layers",
                                    "layers": [128, 64]
                                })
                }
            )
        },

        "rl_trainer": ( RLTrainerComponent, {
            
            "name": "RLTrainerComponent",

            "limit_total_steps": 100_000,

            "predict_optimizations_to_do": True,

            "default_trainer_class": AgentTrainerDQN,

            "agents_trainers_input": {

                "optimization_interval": 512,

                "times_to_learn": 10,

                "batch_size": 128,

                "discount_factor": 0.99,

                "learn_with_all_memory" : False,

                "learning_start_step_delay" : 2000,

                "learner": (

                    DeepQLearnerSchema,

                    {
                        "target_update_rate" : 0.05,

                        "target_update_learn_interval" : 1,

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
                "exploration_strategy" : (EpsilonGreedyStepDecayStrategy,
                                                                  {
                                            "epsilon_end" : 0.04,
                                            "epsilon_start" : 1.0,
                                            "epsilon_decay" :  0.95,
                                            "n_steps_for_decay" : 2000
                                                                  }
                                          ),
                    
                "memory": (

                    TorchMemoryComponent,

                    {
                        "capacity": 5000
                    }

                ),

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
        [10, ["times_to_learn", "optimization_interval", "capacity", "batch_size", "learning_start_step_delay", "n_steps_for_eps_greedy_decay"]],
        [10, ["learning_rate", "clip_grad_strat","target_update_rate", "target_update_learn_interval", "dqn_strategy"]],
        [10, ["policy_layers"]]
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


    return [

        SingleHyperparameterSuggestion(
            name="n_steps_for_eps_greedy_decay",
            value_suggestion=("int", {"low": 500, "high": 4000}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "exploration_strategy", 1, "n_steps_for_decay"]
            ]

        ),

        SingleHyperparameterSuggestion(
            name="dqn_strategy",
            value_suggestion=("cat", {"choices": [DeepQLearnerSchema, DoubleDeepQLearnerSchema]}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "learner", 0]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="times_to_learn",
            value_suggestion=("int", {"low": 2, "high": 20}),
            hyperparameter_localizations=[
                [*agents_trainer_input, "times_to_learn"]
            ]
        ),


        SingleHyperparameterSuggestion(
            name="optimization_interval",
            value_suggestion=("int", {"low": 256, "high": 4096, "step" : 64}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","optimization_interval"],
            ]
        ),

        SingleHyperparameterSuggestion(
            name="capacity",
            value_suggestion=("int", {"low": 5000, "high": 10000}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","memory",1,"capacity"],
            ]
        ),

        SingleHyperparameterSuggestion(
            name="batch_size",
            value_suggestion=("cat", {"choices": [64,128, 256]}),
            hyperparameter_localizations=[
                ["input","rl_trainer",1,"agents_trainers_input","batch_size"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="learning_start_step_delay",
            value_suggestion=("int", {"low": 2000, "high" : 5000}),
            hyperparameter_localizations=[
                [*agents_trainer_input,"learning_start_step_delay"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="target_update_rate",
            value_suggestion=("float", {"low" : 0.025, "high" : 0.99}),
            hyperparameter_localizations=[
                [*learner_input, "target_update_rate"]
            ]
        ),

        SingleHyperparameterSuggestion(
            name="target_update_learn_interval",
            value_suggestion=("int", {"low" : 1, "high" : 20}),
            hyperparameter_localizations=[
                [*learner_input, "target_update_learn_interval"]
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
            name="policy_layers",
            min_len=1,
            max_len=4,
            hyperparameter_localizations=[
                [
                    *policy_model_input, "layers"
                ]
            ],
            hyperparameter_suggestion_for_list=
                SingleHyperparameterSuggestion(
                    value_suggestion=("cat",{"choices":[32,64,128,256]})
                )
        )

    ]