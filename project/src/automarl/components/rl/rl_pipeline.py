import os
import shutil
from automarl.components.basic_components.evaluator_component import ComponentWithEvaluator
from automarl.components.basic_components.exec_component import ExecComponent
from automarl.components.basic_components.seeded_component import SeededComponent
from automarl.component import ParameterSignature, requires_input_process
from automarl.core.advanced_component_creation import get_sub_class_with_correct_parameter_signature
from automarl.components.loggers.component_with_results import ComponentWithResults
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.trainers.rl_trainer_component import RLTrainerComponent
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent
from automarl.components.rl.environment.pettingzoo.aec_pettingzoo_env import AECPettingZooEnvironmentWrapper
from automarl.utils.files_utils import open_or_create_folder
from automarl.components.basic_components.state_management import StatefulComponent

from automarl.components.loggers.result_logger import ResultLogger
import torch

import gc

from automarl.components.loggers.logger_component import LoggerSchema, ComponentWithLogging

from automarl.core.advanced_input_management import ComponentDictParameterSignature, ComponentParameterSignature

from automarl.utils.random_utils import generate_seed, do_full_setup_of_single_seed
from automarl.core.exceptions import common_exception_handling

class RLPipelineComponent(ExecComponent, StatefulComponent, ComponentWithEvaluator, SeededComponent, ComponentWithLogging, ComponentWithResults):
    
    '''
    This component represents a whole RL training process, from the setup of the agents and the environment, to training, to the evaluation of results
    '''

    parameters_signature = {
        
                        "device" : ParameterSignature(default_value="cuda", ignore_at_serialization=True),
                                                                               
                       "environment" : ComponentParameterSignature(default_component_definition=(AECPettingZooEnvironmentWrapper, {}), possible_types=[AECEnvironmentComponent]),
                       
                       "agents" : ComponentDictParameterSignature(default_component_definition={}),
                       "agents_input" : ParameterSignature(default_value={}, ignore_at_serialization=True),
                       
                       "rl_trainer" : ComponentParameterSignature(default_component_definition=(RLTrainerComponent, {})),

                       "fraction_of_training_to_do_in_session" : ParameterSignature(mandatory=False, description="If when this is run it is supposed to do only a fraction of the training, this affects the stop condition"),
                       "generate_fraction_from_times_to_run" : ParameterSignature(default_value=False),

                       "save_checkpoints" : ParameterSignature(default_value=True),

                       "evaluation_report_strategy" : ParameterSignature(default_value="best"),

                       "stop_at_evaluation" : ParameterSignature(mandatory=False)
                
                       }

    exposed_values = {}
    
    
    results_columns = ["times_ran"] # this means a result_logger will exist with the column "episodes_done"

    # INITIALIZATION -----------------------------------------------------------------------------

    def _process_input_internal(self): #this is the best method to have initialization done right after
                
        super()._process_input_internal()

        self.lg.writeLine(f"Processing RL pipeline with values {self.values}\n")
                
        self.device = self.get_input_value("device")

        self.agents_input : dict = self.get_input_value("agents_input")

        state_memory_size = self.agents_input.get("state_memory_size")
        if state_memory_size is not None and state_memory_size <= 1:
            self.lg.writeLine(f"There was state memory defined for agents with size {state_memory_size}, which is less than the minimum of 2")
            self.agents_input.pop("state_memory_size")

        
        self.configure_device(self.device)
        
        self.setup_environment()
                                
        self.initialize_agents_components()
    
        self.setup_trainer()
        
        self.rl_setup_evaluator()
        
        self._setup_checkpoints()

        self.lg.writeLine(f"Finished processing rl pipeline with values {self.values}\n")

        
    def setup_environment(self):
        self.env : AECEnvironmentComponent = self.get_input_value("environment")
                
        self.env.pass_input({"device" : self.device})
        
        self.env.process_input_if_not_processed()
        
        self.lg.writeLine(f"Setting up RL pipeline with environment: {self.env.get_env_name()}")

        self.agents_names_in_environment = self.env.agents()

        self.lg.writeLine(f"Agents in environment: {self.agents_names_in_environment}\n")
    
    
    def rl_setup_evaluator(self):
        
        '''Setups custom logic for '''
        
        if self.component_evaluator is not None:

            self.lg.writeLine(f"Evaluator exists, getting its metrics...")
                
            evaluation_columns = self.component_evaluator.get_metrics_strings()

            self.lg.writeLine(f"Evaluator gives metrics: {evaluation_columns}")
        
            self.add_to_columns_of_results_logger(evaluation_columns)

        self.stop_at_evaluation = self.get_input_value("stop_at_evaluation")

        if self.stop_at_evaluation is not None:
            self.lg.writeLine(f"The rl pipeline will be considered over when it gets an evaluation of {self.stop_at_evaluation}")
    
        
    def configure_device(self, str_device_str):
        
        '''Configures the torch device'''
        
        try:

            self.lg.writeLine("Trying to use cuda...")
            self.device = torch.device(str_device_str)
    
        except Exception as e:
            self.device = torch.device("cpu")
            self.lg.writeLine(f"There was an error trying to setup the device in '{str_device_str}': {str(e)}")

        self.lg.writeLine(f"The device used will be: {self.device}\n")
        
        
    def setup_trainer(self):    
        
        '''Setups the trainer, which is rensonsible for executing the training algorithm, agnostic to the learning methods, models and number of agents'''
                
        rl_trainer_input = {
            "device" : self.device,
            "environment" : self.env,
            "times_to_run" : self._times_to_run,
            "create_new_directory" : False,
            "agents" : self.agents.copy() # so changes to the rl trainer dict do not translate to the agents passed
        }        
        
        self.fraction_of_training_to_do_in_session = self.get_input_value("fraction_of_training_to_do_in_session")

        self.generate_fraction_from_times_to_run = self.get_input_value("generate_fraction_from_times_to_run")

        # if we are to generate a fraction and there was none specified
        if self.generate_fraction_from_times_to_run and self.fraction_of_training_to_do_in_session is None and self._times_to_run is not None: 
            self.fraction_of_training_to_do_in_session = 1 / self._times_to_run


        if self.fraction_of_training_to_do_in_session is not None:
            rl_trainer_input["fraction_training_to_do"] = self.fraction_of_training_to_do_in_session
            self.lg.writeLine(f"Fraction of training to do: {self.fraction_of_training_to_do_in_session} was passed to rl trainer")


        self.rl_trainer : RLTrainerComponent = self.get_input_value("rl_trainer", look_in_value_with_key="rl_trainer", look_in_attribute_with_name="rl_trainer")
        
        self.rl_trainer.pass_input(rl_trainer_input)

            
    def initialize_agents_components(self):
        
        '''Initialize the agents, creating them if necessary first'''

        self.agents : dict[str, AgentSchema] = self.get_input_value("agents", look_in_attribute_with_name="agents") #this is a dictionary with {agentName -> AgentSchema}, the environment must be able to return the agent name

        for agent_name, agent in self.agents.items():
            self.configure_exisent_agent_component(agent_name, agent)

        self.create_agents()
        
        
        for agent_name, agent in self.agents.items():
            self.configure_agent_component(agent_name, agent)

    
        self.input.pop("agents_input", None)
        self.input["agents"] = self.agents

    
    @requires_input_process
    def get_agents(self):

        '''
        Returns the agents dictionary
        '''

        return {**self.agents}

                                                                                    
    def configure_exisent_agent_component(self, agent_name, agent : AgentSchema):
        '''Configures agents that were not created by the RL Pipeline'''
            



    def configure_agent_component(self, agent_name, agent : AgentSchema):
        
        '''Configures the agents, setting up their action and state spaces, the logger and more'''
            
        self.setup_agent_state_action_shape(agent_name, agent)


            
            
    def setup_agent_state_action_shape(self, agent_name, agent : AgentSchema):       
        
        '''Setups the agent state space and action shape'''
         
        state_shape = self.env.get_agent_state_space(agent_name)
                
        self.lg.writeLine(f"State for agent {agent.name} has shape state: {state_shape}")

        action_shape = self.env.get_agent_action_space(agent_name)

        self.lg.writeLine(f"Action space of agent {agent.name} has shape: {action_shape}\n")

        agent.pass_input({"state_shape" : state_shape })
        agent.pass_input({"action_shape" : action_shape })
            

    def __create_agent_name(self, base_name : str, created_names : list):

        if not base_name in created_names:
            return base_name
        
        i = 0
        name_to_try = base_name

        while True:
            name_to_try = f"{base_name}_{i}"
            if not name_to_try in created_names:
                return name_to_try
            i += 1

            
            
    def create_agents(self):
        
        '''Creates agents that did not exist'''

        agents_to_create = [*self.agents_names_in_environment]
        for agent_name in self.agents.keys():
            agents_to_create.pop(agents_to_create.index(agent_name))

        if len(agents_to_create) > 0:
            self.lg.writeLine(f"Agents to create (not passed in input): {agents_to_create}")

        for agent_name in agents_to_create: #worth remembering that the order of the initialization of the agents is defined by the environment

            agent_input = {} #the input of the agent, with base values defined by "agents_input"

            agent_name = self.__create_agent_name(agent_name, self.agents.keys())
            agent_input["name"] = agent_name
            
            agent_input["base_directory"] = os.path.join(self.get_artifact_directory(), "agents" )
            
            agent_class = get_sub_class_with_correct_parameter_signature(AgentSchema, self.agents_input) #gets the agent class with the correct parameter signature

            self.agents[agent_name] = self.initialize_child_component(agent_class, input=agent_input)

            self.lg.writeLine("Created agent in training " + agent_name + " with base directory " + self.agents[agent_name].get_base_directory() + '\n')

            self.agents[agent_name].pass_input(self.agents_input)


    
    def _setup_checkpoints(self):

        self.save_checkpoints = self.get_input_value("save_checkpoints")
        self.evaluation_report_strategy = self.get_input_value("evaluation_report_strategy")

        if self.save_checkpoints is not None and self.save_checkpoints != False:

            self.lg.writeLine(f"Will save checkpoints for evaluations")

            checkpoints_and_evaluations = self.values.get("checkpoints_and_evaluations")

            if checkpoints_and_evaluations is None:
                self.values["checkpoints_and_evaluations"] = []

            if self.evaluation_report_strategy == 'best':
                self.lg.writeLine(f"As the evaluation to report is the best, will save only the best checkpoint")

            else:
                self.lg.writeLine(f"As the evaluation to report is the last, will save all checkpoints")

    
    def _create_checkpoint(self, checkpoint_path):
            
            this_component_path = self.get_artifact_directory()

            if os.path.exists(checkpoint_path):
                self.lg.writeLine(f"Checkpoint already existed, deleting it...")
                shutil.rmtree(checkpoint_path)

            # ensure checkpoint directory exists
            open_or_create_folder(checkpoint_path, create_new=False)

            shutil.copytree(
                this_component_path,
                checkpoint_path,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__checkpoints", "__checkpoint", "__pycache__", "__temp*", "*.tmp")
            )



    def _add_checkpoint(self, evaluation_results):

            '''Adds a checkpoint for the training'''

            checkpoints_and_evaluations : list[tuple[str, float]] = self.values.get("checkpoints_and_evaluations")

            this_component_path = self.get_artifact_directory()

            checkpoint_path = os.path.join(this_component_path, "__checkpoints", str(len(checkpoints_and_evaluations)))

            self.lg.writeLine(f"New checkpoint will be in {checkpoint_path}")

            self._create_checkpoint(checkpoint_path)

            self.lg.writeLine(f"Finished copying component to checkpoint")

            checkpoints_and_evaluations.append((checkpoint_path, evaluation_results))
    
    
    def _create_best_checkpoint(self, evaluation_results):
            
            '''
            Sets the best checkpoint
            '''

            checkpoints_and_evaluations : list[tuple[str, float]] = self.values.get("checkpoints_and_evaluations")

            self.lg.writeLine(f"Current checkpoints are {checkpoints_and_evaluations}")

            is_best_result = False

            last_eval_tuple = self.get_best_evaluation_checkpoint_path_result()

            if last_eval_tuple is None:
                last_max_evaluation = None
            
            else:
                (_, last_max_evaluation) = last_eval_tuple

            if last_max_evaluation is not None:
                if last_max_evaluation["result"] <= evaluation_results["result"]:
                    is_best_result = True
            
            else:
                is_best_result = True

            if is_best_result:

                this_component_path = self.get_artifact_directory()

                checkpoint_path = os.path.join(this_component_path, "__checkpoint")

                self.lg.writeLine(f"New checkpoint will be in {checkpoint_path}")

                self._create_checkpoint(checkpoint_path)

                self.lg.writeLine(f"Finished copying component to checkpoint")

                self.values["checkpoints_and_evaluations"] = [(checkpoint_path, evaluation_results)]
                        
    
    def _evaluate_this_component(self):

        if not self.save_checkpoints:
            return super()._evaluate_this_component() # by default this will return the last evaluation
        
        else:

            self.lg.writeLine(f"Evaluating this component with checkpoints...")

            self.lg.writeLine(f"Will first evaluate this component...")

            evaluation_results = super()._evaluate_this_component()

            self.lg.writeLine(f"Finished evaluating with results: {evaluation_results}")

            self.lg.writeLine(f"Will now save all there is to save into disk...")

            self.save_state_and_rest_to_disk()

            self.lg.writeLine(f"Finished saving to disk")

            if self.evaluation_report_strategy == 'best':
                self._create_best_checkpoint(evaluation_results)

            else:
                self._add_checkpoint(evaluation_results)
            
            checkpoints_and_evaluations : list[tuple[str, float]] = self.values.get("checkpoints_and_evaluations")

            self.lg.writeLine(f"Current checkpoints are: {checkpoints_and_evaluations}")
            for (path, eval_results) in self.values.get("checkpoints_and_evaluations"):
                self.lg.writeLine(f"                    {path} -> {eval_results}")

            if self.evaluation_report_strategy == 'best':

                (max_path, max_evaluation_results) = self.get_best_evaluation_checkpoint_path_result()

                self.lg.writeLine(f"Current best result is {max_evaluation_results} from path {max_path}, returning that one\n")

                return max_evaluation_results
            
            else:
                self.lg.writeLine(f"Returning produced results\n")

                return evaluation_results


    def get_best_evaluation_checkpoint_path_result(self):

        checkpoints_and_evaluations = self.values.get("checkpoints_and_evaluations")

        if checkpoints_and_evaluations is None or len(checkpoints_and_evaluations) == 0:
            return None
        
        (path, evaluation_results) = checkpoints_and_evaluations[0]

        max_evaluation_results = evaluation_results
        max_path = path
        
        for (path, evaluation_results) in checkpoints_and_evaluations[1:]:

            if evaluation_results["result"] >= max_evaluation_results["result"]:
                max_evaluation_results = evaluation_results
                max_path = path

        return (max_path, max_evaluation_results)

    
    def get_best_evaluation_checkpoint_path(self):

        (max_path, max_result) = self.get_best_evaluation_checkpoint_path_result()

        return max_path
    

    def get_best_evaluation_checkpoint_result(self):

        (max_path, max_result) = self.get_best_evaluation_checkpoint_path_result()

        return max_result
        
    # TRAINING_process ----------------------
    
    def _deal_with_exception(self, exception : Exception):
        
        super()._deal_with_exception(exception)

        try:
            self.lg.writeLine(f"RL pipeline had an exception, current state is {self.values.get('running_state', None)}")
        except:
            pass
        
        common_exception_handling(self.lg, exception, 'error_report.txt')
    
        
    @requires_input_process
    def train(self):        
        '''Executes the training part of the algorithm with a specified number of episodes < than the number specified'''
        
        self.lg.writeLine(f"Initiating training process with rl trainer {self.rl_trainer.name}")

        gc.collect() #this forces the garbage collector to collect any abandoned objects
        torch.cuda.empty_cache() #this clears cache of cuda
        
        self.rl_trainer.run()

        self.lg.writeLine(f"Finished training process\n")
        

    def _is_over(self):
        isover = super()._is_over()

        if not isover:
            is_rl_trainer_over = self.rl_trainer.is_over()

            if is_rl_trainer_over:
                self.lg.writeLine(f"As rl trainer defined its running state as over, the rl pipeline will also be considered over")
                isover = is_rl_trainer_over

            if not isover and self.stop_at_evaluation is not None:
                
                last_result = self.get_last_evaluation()

                if last_result is not None and last_result["result"] >= self.stop_at_evaluation:
                    self.lg.writeLine(f"RL pipeline noticed that its last evaluation results ({last_result}) is higher than the value defined to stop it ({self.stop_at_evaluation})")
                    isover = True
        
        return isover

    @requires_input_process
    def _algorithm(self):
        
        '''
        Executes the training part of the algorithm with a specified number of episodes < than the number specified
        
        It then evaluates and returns the results
        '''

        self.lg.writeLine(f"Running RL Pipeline...\n")
        
        self.train() #trains the agents in the reinforcement learning pipeline
        
    

    def pass_evaluation(self, evaluation_results):
        '''
        Passes an evaluation of this component to it for internal processing, such as verifying if the algorithm is over
        '''

        self.log_results({
        "times_ran" : self.values["times_ran"] ,
                          **evaluation_results})
            
        self._test_to_see_if_algorithm_is_over()
        
    
    def _pos_algorithm(self):
        super()._pos_algorithm()

        if self.component_evaluator is not None:
            
            self.lg.writeLine("Evaluating the trained agents...")
            
            evaluation_results = self.evaluate_this_component() # evaluates the resulting model and saves the results

            self.pass_evaluation(evaluation_results)

        else:
            self.log_results({
                "times_ran" : self.values["times_ran"],
            })
        
    
    @requires_input_process
    def get_env(self):
        return self.env
        
    # RESULTS --------------------------------------
    
    @requires_input_process
    def get_results_logger(self) -> ResultLogger:
        return self.rl_trainer.get_results_logger()
    
    # EXPAND TRAINING ------------------------------

    @requires_input_process
    def expand_training(self, multiplier : float) -> None:
        '''Expands the training that will be done in this RL Pipeline using some multiplier'''

        self.input["times_to_run"] *= multiplier
        self._times_to_run *= multiplier

        self.rl_trainer.pass_input({"times_to_run" : self._times_to_run}) 
        self.rl_trainer.expand_training(multiplier)

    
    
    #@requires_input_process    
    #def get_last_Results(self):
    #    
    #    return self.rl_trainer.get_last_results()