from automarl.component import Component
from automarl.components.rl.environment.environment_components import EnvironmentComponent


class RewardFun(Component):

    parameters_signature = {}


    def __init__(self, input = None):
        super().__init__(input)


    def generate_rewards(self, environment : EnvironmentComponent) -> dict[str, float]:
        return {}