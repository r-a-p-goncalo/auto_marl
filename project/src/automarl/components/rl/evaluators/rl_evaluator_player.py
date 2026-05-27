
import os

from automarl.utils.random_utils import generate_seed
import pandas
from automarl.component import Component
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.utils.json_utils.json_component_utils import gen_component_from
from automarl.components.loggers.logger_component import LoggerSchema, use_logger
from automarl.components.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.components.loggers.result_logger import aggregate_results_logger

from automarl.core.input_management import ParameterSignature

from automarl.components.rl.rl_player.rl_player import RLPlayer
from automarl.components.rl.evaluators.rl_std_avg_evaluator import LastValuesAvgStdEvaluator
from automarl.utils.files_utils import loadDataframe, saveDataframe
from automarl.utils.configuration_component_utils import save_configuration
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.environment.environment_components import EnvironmentSampler

class EvaluatorWithPlayer(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines, which evaluates the last results it has and uses them to compute a result, penalizing high variance and using the mean as the base value
    
    This is meant to be used not as a final evaluation of a component, but as an intermediary evaluator at training time.
    
    '''
    
    parameters_signature = {
        "base_evaluator" : ComponentParameterSignature(default_component_definition=(LastValuesAvgStdEvaluator, {})),
        "rl_player_definition" : ParameterSignature(default_value=(RLPlayer, {})),
        "number_of_episodes" : ParameterSignature(default_value=5),
        "number_of_evaluations" : ParameterSignature(default_value=1),
        "environment" : ComponentParameterSignature(mandatory=False),
        "save_after_evaluation" : ParameterSignature(ignore_at_serialization=True, default_value=False),
        "setup_seeds_of_player" : ParameterSignature(default_value=True),
        "directory_to_store_evaluation_prefix" : ParameterSignature(default_value="evaluations"),
    }
    
    exposed_values = {
        
        "last_evaluation" : {},
        "number_of_evaluations" : 0
    
    }
    

    def _process_input_internal(self):
        
        super()._process_input_internal()
        
        self.base_evaluator : RLPipelineEvaluator = self.get_input_value("base_evaluator")

        self.number_of_episodes = self.get_input_value("number_of_episodes")
        self.number_of_evaluations = self.get_input_value("number_of_evaluations")
        self.rl_player_definition = self.get_input_value("rl_player_definition")

        self.save_after_evaluation = self.get_input_value("save_after_evaluation")
        
        self._setup_environment()

        self.setup_seeds_of_player = self.get_input_value("setup_seeds_of_player")

        if isinstance(self.setup_seeds_of_player, list):
            pass

        elif self.setup_seeds_of_player:
            self.setup_seeds_of_player = [generate_seed() for _ in range(self.number_of_evaluations)]
            self.input["setup_seeds_of_player"] = self.setup_seeds_of_player

        else:
            self.setup_seeds_of_player = None
        
        self.directory_to_store_evaluation_prefix = self.get_input_value("directory_to_store_evaluation_prefix")



    def _setup_environment(self):
        
        self.env = None
        
        if "environment" in self.input.keys():
            self.env = self.get_input_value("environment")
            
            
    # EVALUATION -------------------------------------------------------------------------------
    
    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), *self.base_evaluator.get_metrics_strings()]
    
    def _evaluate(self, component_to_evaluate : RLPipelineComponent):

        if isinstance(component_to_evaluate, tuple):
            return self._evaluate_from_tuple(component_to_evaluate)
    
        else: # Is of type RLPipelineComponent or something with same attributes
            return self._evaluate_from_component(component_to_evaluate)
        
    
    
    def _evaluate_from_component(self, component_to_evaluate : RLPipelineComponent):
        
        if self.env is not None:
            env = self.env
        else:
            env = component_to_evaluate.get_env()

        component_to_evaluate.process_input_if_not_processed()

        agents = component_to_evaluate.agents
        device = component_to_evaluate.device
        evaluations_directory = os.path.join(component_to_evaluate.get_artifact_directory(), self.directory_to_store_evaluation_prefix)


        return self._evaluate_agents(agents, device, evaluations_directory, env, component_to_evaluate)
        
        
    def _evaluate_from_tuple(self, tuple):
        
        (agents, device, evaluations_directory, env) = tuple

        return self._evaluate_agents(agents, device, evaluations_directory, env)
    

    def __generalize_get_environment(self, env, rl_player : RLPlayer = None):
        
        '''Generalizes the process of generating, setting or simply using a passed environment'''

        if env is not None:
            env = env # we use the environment passed
            
        elif self.env is not None: 
            env = self.env # we use the environment stored in the component
        
        else:
            return None # no environment was passed or stored, we return None

        env = gen_component_from(env) # in case the environment passed isn't an instance but a definition of an environment

        if isinstance(env, EnvironmentSampler): # if a sampler was passed
            
            env = env.sample()
            
            env.pass_input({
                "base_directory" : rl_player.get_artifact_directory()
            })
        
        return env
    

    def _load_evaluation_result_df(self, evaluations_directory):

        try:
            return loadDataframe(evaluations_directory, "evaluations.csv")
        
        except:
            return None
    
    def _save_evaluation_result(self, result, evaluations_directory, current_daframe : pandas.DataFrame=''):


        if current_daframe == '':
            current_daframe = self._load_evaluation_result_df(evaluations_directory)

        if current_daframe is None: # TODO: add column named "evaluation" which is the number of the row
            saveDataframe(pandas.DataFrame([{"evaluation" : 0, **result}]), evaluations_directory, "evaluations.csv")

        else:
            new_row = pandas.DataFrame([{"evaluation" : len(current_daframe), **result}])

            updated = pandas.concat([current_daframe, new_row], ignore_index=True)
            saveDataframe(updated, evaluations_directory, "evaluations.csv")


        

    def _evaluate_agents(self, agents, device, evaluations_directory, env=None, component_to_evaluate=None):

        '''Evaluate agents using the RL player and the base evaluator'''
        
        results_loggers_of_new_plays = []
                
        # compute new evaluations
        for i in range(self.number_of_evaluations): # evaluate plays and store their paths 

            seed_for_player = None if self.setup_seeds_of_player is None else self.setup_seeds_of_player[i]

            rl_player_of_run : RLPlayer = self._run_play_to_evaluate(agents, device, evaluations_directory, env, seed_for_player, component_to_evaluate) 

            results_loggers_of_new_plays.append(rl_player_of_run.get_results_logger())

        current_eval_dataframe = self._load_evaluation_result_df(evaluations_directory)

        n_eval_done = 0 if current_eval_dataframe is None else len(current_eval_dataframe)

        results_logger_of_new_plays = aggregate_results_logger(results_loggers_of_new_plays, evaluations_directory, new_results_filename=f"evaluation_results_{n_eval_done}.csv")
                
        evaluation_to_return = self.base_evaluator.evaluate(results_logger_of_new_plays)

        self._save_evaluation_result(evaluation_to_return, evaluations_directory)


        return evaluation_to_return
    
    def _initialize_player_to_run(self, agents : dict[str, AgentSchema], device, evaluations_directory, env, seed_for_player=None, component_to_evaluate : RLPipelineComponent = None):

        rl_player : RLPlayer = gen_component_from(self.rl_player_definition)

        rl_player_input = {
            "agents" : agents,
            "num_episodes" : self.number_of_episodes,
            "base_directory" : evaluations_directory,
            "artifact_relative_directory" : "evaluation",
            "create_new_directory" : True # if there were other play made, create a new one
        }

        if seed_for_player is not None:
            rl_player_input["seed"] = seed_for_player

        rl_player.pass_input(rl_player_input)
        
        env = self.__generalize_get_environment(env, rl_player)
        
        if env is not None:
            rl_player.pass_input({"environment" : env})

        return rl_player

    

    def _run_play_to_evaluate(self, agents : dict[str, AgentSchema], device, evaluations_directory, env, seed_for_player=None, component_to_evaluate=None):

        '''
            Plays a session with the RL player, returning it after
            The objective is to then evaluate the results with the evaluator
        '''

        rl_player_will_be_generated = not isinstance(self.rl_player_definition, Component)
                
        rl_player : RLPlayer = self._initialize_player_to_run(agents, device, evaluations_directory, env, seed_for_player, component_to_evaluate)

        evaluation_logger = LoggerSchema(input={"create_new_directory" : False, "base_directory" : rl_player, "artifact_relative_directory" : '', })            
        
        with use_logger(evaluation_logger):
            rl_player.run()

        if rl_player_will_be_generated: # if the player will be generated, might as well save the configuration for later consultation of it
            save_configuration(rl_player, rl_player.get_artifact_directory(), save_exposed_values=True, ignore_defaults=False)
        
        return rl_player