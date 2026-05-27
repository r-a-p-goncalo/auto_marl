


from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.utils.json_utils.json_component_utils import gen_component_from
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent
             

def initialize_agents_components(agents, env : AECEnvironmentComponent, agents_input={}, caller_component=None) -> dict[str, AgentSchema]:
        
        '''Initialize the agents given the specifications'''

        if agents  == {}:
            raise Exception("No agents defined, can't proceed")
        
        elif not isinstance(agents, dict): # if it is a single specification
            
            if env == None:
                raise Exception("Can't assume name of agent without environment specified")

            single_agent = gen_component_from(agents, caller_component)

            agent_name = next(env.agent_iter()) # get next name of agent

            single_agent.pass_input({"name" : agent_name}) # set the name of the agent
            
            agents_to_return = {agent_name : single_agent}
            
            configure_agent_component(agent_name, single_agent, env, agents_input)

            return agents_to_return

        else: # is non empty dict
            for agent_name in agents.keys(): #change the specification given
            
                agent = gen_component_from(agents[agent_name], caller_component) # generate agent if it was not defined
            
                configure_agent_component(agent_name, agent, env, agents_input)                        
                return agents 
        
                               
                                                                                    
            
def configure_agent_component( agent_name, agent : AgentSchema, env, agents_input={}):
        
        '''Configures the agents, setting up their action and state spaces, the logger and more'''
            
        setup_agent_state_action_shape(agent_name, agent, env)
                            
        agent.pass_input(agents_input)
            
            
def setup_agent_state_action_shape( agent_name, agent : AgentSchema, env : AECEnvironmentComponent):       
        
        '''Setups the agent state space and action shape'''
         
        state_shape = env.get_agent_state_space(agent_name)
                
        action_shape = env.get_agent_action_space(agent_name)

        agent.pass_input({"state_shape" : state_shape })
        agent.pass_input({"action_shape" : action_shape })