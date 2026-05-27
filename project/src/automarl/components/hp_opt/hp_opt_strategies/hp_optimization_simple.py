
from automarl.components.hp_opt.hp_optimization_pipeline import Component_to_opt_type, HyperparameterOptimizationPipeline

import optuna

class SimpleHyperparameterOptimizationPipeline(HyperparameterOptimizationPipeline):

    '''
    A simple Hyperparameter Optimization pipeline, which supports remembering a single configuration
    '''
    
    parameters_signature = {
                                              
                       }
            

    # INITIALIZATION -----------------------------------------------------------------------------


    def _process_input_internal(self): # this is the best method to have initialization done right after
                
        super()._process_input_internal()
                
        self.component_being_optimized : Component_to_opt_type = None
                
    

    # OPTIMIZATION -------------------------------------------------------------------------
    
    def get_component_to_test_path(self, trial : optuna.Trial) -> str:

        if self.component_being_optimized is None:
            self.component_being_optimized = self._create_component_to_optimize(trial) 

        return self.component_being_optimized.get_artifact_directory()

    def get_component_to_test(self, trial : optuna.Trial) -> Component_to_opt_type:
        
        if self.component_being_optimized is None:
            self.component_being_optimized = self._create_component_to_optimize(trial) 
        
        return self.component_being_optimized
    
    def after_trial(self, study, trial):
        super().after_trial(study, trial)

        self.component_being_optimized = None

