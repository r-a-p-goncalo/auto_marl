

from automarl.components.loggers.logger_component import ComponentWithLogging


class HiddenStateSampler(ComponentWithLogging):
    def sample(self, env, agent_name, information_state, n_samples):
        """
        Return complete true states consistent with what agent_name knows.
        """
        raise NotImplementedError()