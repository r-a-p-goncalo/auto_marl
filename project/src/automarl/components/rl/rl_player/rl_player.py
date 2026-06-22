import os
import traceback
from automarl.components.basic_components.exec_component import ExecComponent
from automarl.component import ParameterSignature, requires_input_process
from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.loggers.component_with_results import ComponentWithResults
from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.basic_components.state_management import StatefulComponent

from automarl.components.rl.rl_setup_util import initialize_agents_components

from automarl.serialization.configuration_component_utils import save_configuration
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent
import torch


from automarl.components.loggers.logger_component import  ComponentWithLogging

class RLPlayer(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent):
    
    
    parameters_signature = {
                                                                                       
                       "environment" :  ComponentParameterSignature(),
                       "agents" : ParameterSignature(),
                       "agents_input" : ParameterSignature(default_value={}, ignore_at_serialization=True),
                       "num_episodes" : ParameterSignature(default_value=1),
                       "store_env_at_end" : ParameterSignature(default_value=False),
                       "device" : ParameterSignature(default_value="cpu")

                       
                       }
    

    exposed_values = {"total_steps" : 0,
                      "episode_steps" : 0,
                      "episodes_done" : 0,
                      "episode_score" : 0
                      } 
    
    results_columns = ["episode", "episode_reward", "episode_steps", "avg_reward", "environment"]
    
    def _process_input_internal(self): #this is the best method to have initialization done right after
        
        super()._process_input_internal()

        self.env : AECEnvironmentComponent = self.get_input_value("environment")
        self.num_episodes = self.get_input_value("num_episodes")
        
        self.store_env_at_end = self.get_input_value("store_env_at_end")

        self._setup_agents()

        
    def _setup_agents(self):

        self.agents = self.get_input_value("agents")
        self.agents_input = self.get_input_value("agents_input")
        
        self.agents : dict[str, AgentSchema] = initialize_agents_components(self.agents, self.env, self.agents_input, self)

        self.values["agents_episode_score"] = {agent.name : 0 for agent in self.agents.values()}

        self.add_to_columns_of_results_logger([f"{agent_name}_reward" for agent_name in self.values["agents_episode_score"].keys()])

        # if the agents have no base directory associated with it, use RL player's
        for agent in self.agents.values():
            if not "base_directory" in agent.input.keys():
                self.lg.writeLine(f"Agent {agent.name} has no base directory, passing player's directory to it")
                agent.pass_input({"base_directory" : self.get_artifact_directory()})
        
        
    
    def _setup_episode(self):

        self.values["episode_steps"] = 0
        self.values["episode_score"] = 0
        
        for agent_name in self.values["agents_episode_score"].keys():
            self.values["agents_episode_score"][agent_name] = 0

        self.env.reset()
        
                
        for agent in self.agents.values():
            agent.reset_agent_in_environment(self.env.observe(agent.name))



    def _end_episode(self):
        self.values["episodes_done"] = self.values["episodes_done"] + 1
                
        self.calculate_and_log_results()


    def _run_episode(self):
        '''
        Internal function that runs the the current episode
        '''  

            
    def _do_agent_step(self, agent_name):
       '''
        Internal function that makes an agent act
        '''  
    
    
    @requires_input_process
    def play(self):
        
        for ep in range(self.num_episodes):

            self._setup_episode()

            self._run_episode()

            self._end_episode()
            
        if self.store_env_at_end:
            save_configuration(self.env, self.get_artifact_directory(), "env_config.json", save_exposed_values=True, ignore_defaults=False)
            
                    
        self.env.close()


    # RESULTS LOGGING --------------------------------------------------------------------------------
    
    def calculate_results(self):
                
        return {
            "episode" : [self.values["episodes_done"]],
            "episode_reward" : [float(self.values["episode_score"])],
            "episode_steps" : [self.values["episode_steps"]], 
            "avg_reward" : [float(self.values["episode_score"]) / self.values["episode_steps"]] if self.values["episode_steps"] > 0 else [0.0],
            "environment" : [self.env.name],
            **{f"{agent_name}_reward" : [agent_reward] for agent_name, agent_reward in self.values["agents_episode_score"].items()}
            }
        



    def _algorithm(self):
        self.play()        
        
    

        
