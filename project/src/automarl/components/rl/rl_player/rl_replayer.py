import os
import traceback
from automarl.components.basic_components.evaluator_component import ComponentWithEvaluator
from automarl.components.basic_components.exec_component import ExecComponent
from automarl.component import ParameterSignature, Component, requires_input_process
from automarl.core.advanced_component_creation import get_sub_class_with_correct_parameter_signature
from automarl.core.advanced_input_management import ComponentDictParameterSignature, ComponentParameterSignature, ComponentListParameterSignature
from automarl.components.loggers.component_with_results import ComponentWithResults
from automarl.components.ml.memory.memory_components import MemoryComponent
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.ml.optimizers.optimizer_components import AdamOptimizer
from automarl.components.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automarl.components.rl.trainers.rl_trainer.rl_trainer_component import RLTrainerComponent
from automarl.components.rl.environment.environment_components import AECEnvironmentComponent
from automarl.components.rl.environment.pettingzoo.pettingzoo_env import PettingZooEnvironmentWrapper
from automarl.utils.files_utils import open_or_create_folder
from automarl.components.basic_components.state_management import StatefulComponent

import torch

import gc

from automarl.components.loggers.logger_component import LoggerSchema, ComponentWithLogging

from automarl.utils.random_utils import generate_seed, do_full_setup_of_single_seed

class RLRePlayer(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent, ComponentWithEvaluator):
    
    '''A class which replays the actions in memory'''
    
    parameters_signature = {
                                                                                       
                       "environment" :  ComponentParameterSignature(default_component_definition=(PettingZooEnvironmentWrapper, {})),
                       
                       "memory_list" : ComponentDictParameterSignature(mandatory=True)
                       
                       }
    
    
    results_columns = ["episodes_done"] # this means a result_logger will exist with the column "episodes_done"

    # INITIALIZATION -----------------------------------------------------------------------------

    def _process_input_internal(self): #this is the best method to have initialization done right after
        
        super()._process_input_internal()
        
        self.setup_environment()
        self.setup_memories_of_agents()


    def setup_environment(self):
        self.env : AECEnvironmentComponent = self.get_input_value("environment")

    def setup_memories_of_agents(self):        
        self.memory_dict : dict[MemoryComponent] = self.get_input_value("memory_list")



    def play_episode():
        pass

    def _algorithm(self):
        
        super()._algorithm()
        