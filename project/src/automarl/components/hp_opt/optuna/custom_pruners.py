import optuna
from optuna.pruners import BasePruner

class MixturePruner(BasePruner):


    def __init__(
        self,
        pruners: list[BasePruner]
    ):
        self.pruners = pruners

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:

        for pruner in self.pruners:
            if pruner.prune(study, trial):
                return True
            
        return False