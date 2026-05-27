

from automarlcore.advanced_input_management import ComponentParameterSignature
from automarl.components.hp_opt.samplers.sampler import OptunaSamplerComponent, OptunaSamplerWrapper

import optuna

from optuna.samplers import BaseSampler, RandomSampler
from optuna.study import Study
from optuna.trial import FrozenTrial


class AlternatingRandomSampler(OptunaSamplerComponent):

    parameters_signature = {
        "other_sampler" : ComponentParameterSignature(default_component_definition=(OptunaSamplerWrapper, {}))
    }

    def process_input(self):
        super().process_input()

        self.other_sampler : OptunaSamplerComponent = self.get_input_value("other_sampler") 
        self.other_sampler_optuna : BaseSampler = self.other_sampler.get_optuna_sampler()

        self._random_sampler = RandomSampler()


    def use_random(self, trial_number):
        return trial_number % 2 == 0


    def reseed_rng(self) -> None:
        self.other_sampler_optuna.reseed_rng()
        self._random_sampler.reseed_rng()


    def infer_relative_search_space(
        self,
        study: Study,
        trial: FrozenTrial
    ) -> dict[str, optuna.distributions.BaseDistribution]:

        if self.use_random(trial.number):
            return self._random_sampler.infer_relative_search_space(study, trial)
        else:
            return self.other_sampler_optuna.infer_relative_search_space(study, trial)


    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, optuna.distributions.BaseDistribution]
    ) -> dict:

        if self.use_random(trial.number):
            return self._random_sampler.sample_relative(study, trial, search_space)
        else:
            return self.other_sampler_optuna.sample_relative(study, trial, search_space)


    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: optuna.distributions.BaseDistribution,
    ):


        if self.use_random(trial.number):
            value = self._random_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        else:
            value = self.other_sampler_optuna.sample_independent(
                study, trial, param_name, param_distribution
            )

        

        return value
    
    def reseed_rng(self):
        self._random_sampler.reseed_rng()
        self.other_sampler_optuna.reseed_rng()
    

    def before_trial(
        self,
        study: Study,
        trial: FrozenTrial,
    ) -> None:
        self._random_sampler.before_trial(study, trial)
        self.other_sampler_optuna.before_trial(study, trial)


    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: optuna.trial.TrialState,
        values: list[float] = None,
    ) -> None:

        # Forward callback to both samplers
        self._random_sampler.after_trial(study, trial, state, values)
        self.other_sampler_optuna.after_trial(study, trial, state, values)