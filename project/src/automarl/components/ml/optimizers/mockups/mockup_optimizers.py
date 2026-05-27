

from automarl.components.ml.optimizers.optimizer_components import OptimizerSchema


class MockupOptimizerSchema(OptimizerSchema):
        
    def optimize_model(self, predicted, correct) -> None:
        
        pass