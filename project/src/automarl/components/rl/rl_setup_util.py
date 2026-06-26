from automarl.components.rl.agent.agent_components import AgentSchema
from automarl.components.rl.environment.aec_environment import AECEnvironmentComponent
from automarl.serialization.json_component_utils import gen_component_from


def initialize_agents_components(
    agents,
    env: AECEnvironmentComponent,
    agents_input=None,
    caller_component=None,
) -> dict[str, AgentSchema]:
    """Initialize the agents given the specifications."""

    if agents_input is None:
        agents_input = {}

    if agents == {}:
        raise Exception("No agents defined, can't proceed")

    elif not isinstance(agents, dict):
        if env is None:
            raise Exception("Can't assume name of agent without environment specified")

        single_agent = gen_component_from(agents, caller_component)

        agent_name = next(env.agent_iter())
        single_agent.pass_input({"name": agent_name})

        agents_to_return = {agent_name: single_agent}

        configure_agent_component(agent_name, single_agent, env, agents_input)

        return agents_to_return

    else:
        agents_to_return = {}

        for agent_name in agents.keys():
            agent = gen_component_from(agents[agent_name], caller_component)

            configure_agent_component(agent_name, agent, env, agents_input)
            agents_to_return[agent_name] = agent

        return agents_to_return


def configure_agent_component(
    agent_name,
    agent: AgentSchema,
    env,
    agents_input=None,
):
    """Configure the agent state/action spaces and extra shared input."""

    if agents_input is None:
        agents_input = {}

    setup_agent_state_action_shape(agent_name, agent, env)
    agent.pass_input(agents_input)


def setup_agent_state_action_shape(
    agent_name,
    agent: AgentSchema,
    env: AECEnvironmentComponent,
):
    """Set up the agent state space and action shape."""

    state_shape = env.get_agent_state_space(agent_name)
    action_shape = env.get_agent_action_space(agent_name)

    agent.pass_input({"state_shape": state_shape})
    agent.pass_input({"action_shape": action_shape})