from automarl.components.hp_opt.hyperparameter_suggestion import HyperparameterSuggestion
import optuna
from automarl.components.hp_opt.hp_optimization_pipeline import HyperparameterOptimizationPipeline


RL_PIPELINE = ([])

TRAINER_INPUT = ["rl_trainer" , 1]

AGENTS_INPUT = ["agents_input"]

POLICY_INPUT = [*AGENTS_INPUT, "policy", 1]

MODEL_INPUT = [*POLICY_INPUT, "model", 1]

AGENTS_TRAINER_INPUT = [*TRAINER_INPUT, "agents_trainers_input"]

EXPLORATION_STRATEGY_INPUT = [*AGENTS_TRAINER_INPUT, "exploration_strategy", 1]

AGENT_MEMORY_INPUT = [*AGENTS_TRAINER_INPUT, "memory", 1]

AGENT_LEARNER_INPUT = [*AGENTS_TRAINER_INPUT, "learner", 1]

LEARNER_OPTIMIZER_INPUT = [*AGENT_LEARNER_INPUT, "optimizer", 1]




def get_hyperparameters_to_change() -> list[HyperparameterSuggestion]:
    
    hyperparameters_to_change : list[HyperparameterSuggestion] = [] 
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                                
                            HyperparameterSuggestion(
                                name='discount_factor', 
                                hyperparameter_localizations= [
                                    ([], [*AGENTS_TRAINER_INPUT, "discount_factor"])
                                ],
                                value_suggestion = ('float', {'low':0.5, 'high':0.99}) 
                            )
                            ]       
    
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 

                                 HyperparameterSuggestion(
                                    name='epsilon_start', 
                                    hyperparameter_localizations= [
                                        ([], [*EXPLORATION_STRATEGY_INPUT, "epsilon_start"])
                                    ],
                                    value_suggestion = ('float', {'low':0.95, 'high':0.999}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='epsilon_end', 
                                    hyperparameter_localizations= [
                                        ([], [*EXPLORATION_STRATEGY_INPUT, "epsilon_end"])
                                    ],
                                    value_suggestion = ('float', {'low':0.05, 'high':0.3}) 
                                ),
                                 HyperparameterSuggestion(
                                    name='epsilon_decay', 
                                    hyperparameter_localizations= [
                                        ([], [*EXPLORATION_STRATEGY_INPUT, "epsilon_decay"])
                                    ],
                                    value_suggestion = ('float', {'low':0.95, 'high':0.9999}) 
                                 )

    ]
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                             
                             HyperparameterSuggestion(
                                name='hidden_layers', 
                                hyperparameter_localizations= [
                                    ([], [*MODEL_INPUT, "hidden_layers"])
                                ],
                                value_suggestion = ('int', {'low':2, 'high':8}) 
                            ),
                             HyperparameterSuggestion(
                                name='hidden_size', 
                                hyperparameter_localizations= [
                                    ([], [*MODEL_INPUT, "hidden_size"])
                                ],
                                value_suggestion = ('cat', {'choices' : [16, 32, 64, 128, 256]}) 
                            )
    ]
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                             
                             HyperparameterSuggestion(
                                name='target_update_rate', 
                                hyperparameter_localizations= [
                                    ([], [*AGENT_LEARNER_INPUT, "target_update_rate"])
                                ],
                                value_suggestion = ('float', {'low':0.01, 'high':0.15}) 
                            ),
                             HyperparameterSuggestion(
                                name='learning_rate', 
                                hyperparameter_localizations= [
                                    ([], [*LEARNER_OPTIMIZER_INPUT, "learning_rate"])
                                ],
                                value_suggestion = ('float', {'low':0.000001, 'high':0.1}) 
                            )            
    ]
    
    hyperparameters_to_change = [ *hyperparameters_to_change, 
                             
                             HyperparameterSuggestion(
                                name='memory_capacity', 
                                hyperparameter_localizations= [
                                    ([], [*AGENT_MEMORY_INPUT, "capacity"])
                                ],
                                value_suggestion = ('int', {'low':100, 'high':1000}) 
                            )
                           
    
    ]
    
    return hyperparameters_to_change

def get_configuration_dict(*args, mockup, **kwargs):
    
    if mockup:
        return load_configuration_dict('mockup_basic_dqn', *args, **kwargs)
    
    else:
        return load_configuration_dict('basic_dqn', *args, **kwargs)


def gen_hp_optimization_input(hyperparameters_to_change, configuration_dict=None, num_trials=20, directory='data\\experiments', evaluator=None, mockup=False):

    hp_opt_input = {
    "hyperparameters_range_list" : hyperparameters_to_change,
    "n_trials" : num_trials,
    "steps" : 2,
    "pruner" : optuna.pruners.PercentilePruner(percentile=25.0),
    "base_directory" :  directory,
    }
    
    if configuration_dict != None:
        hp_opt_input["configuration_dict"] = configuration_dict
    
    if evaluator is not None:
        hp_opt_input["evaluator_component"] = evaluator
        
    if mockup:
        
        from automarl.components.basic_components.mock_components.mock_evaluators import RandomMockEvaluator
        hp_opt_input["evaluator_component"] = (RandomMockEvaluator, {})
    
    return hp_opt_input
    
    
    
def main(num_episodes, num_trials, directory, mockup):
    
    if mockup:
        print("Running mockup test")
    
    hyperparameters_to_change = get_hyperparameters_to_change()
    
    print("Generating hyperparameters to change")

    configuration_dict = get_configuration_dict(num_episodes=num_episodes, mockup=mockup)

    hp_opt_input = gen_hp_optimization_input(hyperparameters_to_change, configuration_dict, num_trials=num_trials, directory=directory, mockup=mockup)

    hp_opt_pipeline = HyperparameterOptimizationPipeline(hp_opt_input)
    
    hp_opt_pipeline.run()
    
    #hp_opt_pipeline.save_configuration()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization pipeline.")
    parser.add_argument("--num_episodes", type=int, default=12, help="Number of episodes to run.")
    parser.add_argument("--num_trials", type=int, default=20, help="Number of trials to run.")
    parser.add_argument("--directory", type=str, default='.\\data\\experiments', help="Directory to save results.")
    parser.add_argument("--mockup", type=bool, default=False, help="Configuration to use for the experiment.")

    args = parser.parse_args()

    main(num_episodes=args.num_episodes, num_trials=args.num_trials, directory=args.directory, mockup=args.mockup)