

from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.core.input_management import ParameterSignature


class ImperfectInformationSearchPlanner(ComponentWithLogging):

    parameters_signature = {
    
        "num_simulations": ParameterSignature(default_value=200),
        "c_puct": ParameterSignature(default_value=1.5),
        "temperature": ParameterSignature(default_value=1.0),
        "hidden_state_samples": ParameterSignature(default_value=1),
        "use_public_tree": ParameterSignature(default_value=True),
    
    }

    def search(
        self,
        env,
        agent_name,
        information_state,
        evaluator,
        opponent_policy=None,
        training=True,
    ):
        """
        Returns:
            action
            search_policy
            root_value
            visit_counts
            sampled_hidden_states_info
        """