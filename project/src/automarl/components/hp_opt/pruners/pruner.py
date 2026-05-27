

from automarl.components.basic_components.seeded_component import SeededComponent
from automarl.component import requires_input_process
from automarlcore.input_management import ParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging

import optuna
from automarl.components.hp_opt.optuna.custom_pruners import MixturePruner

class OptunaPrunerComponent(ComponentWithLogging):

    @requires_input_process
    def get_optuna_pruner(self) -> optuna.pruners.BasePruner:
        """Returns an optuna pruner instance"""
        return self
    

class OptunaPrunerWrapper(OptunaPrunerComponent, SeededComponent):

    parameters_signature = {
        "optuna_pruner": ParameterSignature(default_value="Median"),
        "pruner_input": ParameterSignature(mandatory=False)
    }

    def process_input(self):

        super().process_input()

        self._initialize_pruner()

    def _initialize_pruner(self):

        self.optuna_pruner = self.get_input_value("optuna_pruner")

        if isinstance(self.optuna_pruner, str):
            self._initialize_pruner_from_str(self.optuna_pruner)

        elif isinstance(self.optuna_pruner, type):
            self._initialize_pruner_from_class(self.optuna_pruner)

        else:
            raise Exception("Invalid pruner type")

    def _initialize_pruner_from_class(self):

        pruner_input = self.get_input_value("pruner_input") or {}

        self.optuna_pruner = self.optuna_pruner(**pruner_input)

    def _initialize_pruner_from_str(self, pruner_str):

        pruner_input = self.get_input_value("pruner_input") or {}

        if pruner_str == "Median":

            self.optuna_pruner = optuna.pruners.MedianPruner(**pruner_input)

        elif pruner_str == "Percentile":

            self.optuna_pruner = optuna.pruners.PercentilePruner(**pruner_input)

        elif pruner_str == "Hyperband":

            self.optuna_pruner = optuna.pruners.HyperbandPruner(**pruner_input)

        elif pruner_str == "MixturePruner":

            pruners = []

            for pruner_def in pruner_input["pruners"]:

                pruner_class, pruner_params = pruner_def

                if isinstance(pruner_class, str):
                    pruner = OptunaPrunerWrapper(
                        optuna_pruner=pruner_class,
                        pruner_input=pruner_params
                    ).get_optuna_pruner()
                else:
                    pruner = pruner_class(**pruner_params)

                pruners.append(pruner)

            self.optuna_pruner = MixturePruner(pruners)

        else:
            raise NotImplementedError(f"Invalid pruner '{pruner_str}'")

    @requires_input_process
    def get_optuna_pruner(self) -> optuna.pruners.BasePruner:

        return self.optuna_pruner