"""Small PPO experiment for the AEC Nim toy environment."""

from automarl.components.fundamentals.translator.tensor_translator import ToTorchTranslator
from automarl.components.ml.memory.torch_memory_component import TorchMemoryComponent
from automarl.components.ml.models.neural_model import FullyConnectedModelSchema
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.rl.environment.adversarial.nim_aec_env import NimAECEnvironment
from automarl.components.rl.learners.ppo_learner import PPOLearner
from automarl.components.rl.policy.stochastic_policy import MaskedCategoricalStochasticPolicy
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.rl.trainers.agent_trainer.agent_trainer_ppo import AgentTrainerPPO
from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent
from automarl.components.rl.evaluators.rl_agent_iter_evaluator import RLAgentIterEvaluator
from automarl.components.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator
from automarl.components.rl.evaluators.rl_vs_agents_evaluator import AgentVsAgentsWithPolicy
from automarl.components.rl.policy.random_policy import RandomPolicyMasked
from automarl.components.rl.rl_player.rl_player import RLPlayer


def experiment_name():
    return "nim_ppo"


def config_dict():
    return {
        "__type__": RLPipelineComponent,
        "name": "NimPPOPipeline",
        "input": {
            "device": "cpu",
            "save_checkpoints": False,
            "environment": (
                NimAECEnvironment,
                {
                    "initial_stones": 15,
                    "max_take": 3,
                    "start_player": "player_0",
                },
            ),
            "agents_input": {
                "state_translator": (ToTorchTranslator, {}),
                "policy": (
                    MaskedCategoricalStochasticPolicy,
                    {
                        "model": (
                            FullyConnectedModelSchema,
                            {
                                "layers": [32, 32],
                            },
                        ),
                    },
                ),
            },
            "rl_trainer": (
                RLTrainerComponent,
                {
                    "name": "NimPPOTrainer",
                    "limit_total_steps": 20_000,
                    "predict_optimizations_to_do": False,
                    "default_trainer_class": AgentTrainerPPO,
                    "agents_trainers_input": {
                        "optimization_interval": 256,
                        "times_to_learn": 6,
                        "batch_size": 64,
                        "discount_factor": 0.99,
                        "learn_with_all_memory": True,
                        "learner": (
                            PPOLearner,
                            {
                                "lambda_gae": 0.95,
                                "clip_epsilon": 0.2,
                                "entropy_coef": 0.01,
                                "value_loss_coef": 0.5,
                                "critic_model": (
                                    FullyConnectedModelSchema,
                                    {
                                        "layers": [32, 32],
                                    },
                                ),
                                "optimizer": (
                                    AdamOptimizer,
                                    {
                                        "learning_rate": 3e-4,
                                    },
                                ),
                            },
                        ),
                        "memory": (
                            TorchMemoryComponent,
                            {
                                "capacity": 1024,
                            },
                        ),
                    },
                },
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
