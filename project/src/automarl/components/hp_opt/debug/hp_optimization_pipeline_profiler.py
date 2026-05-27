import cProfile
import os

from automarl.components.hp_opt.hp_optimization_pipeline import HyperparameterOptimizationPipeline
import optuna


class HyperparameterOptimizationPipelineProfiler(HyperparameterOptimizationPipeline):
    
    is_debug_schema = True

    parameters_signature = {
                           
                       }
            

    def objective(self, trial : optuna.Trial):
        
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            to_return = super().objective(trial)

            profiler_output_path = self._get_path_of_component_of_trial(trial)
            profiler_output_path = os.path.join(profiler_output_path, "profiler.out")

            profiler.disable()
            profiler.dump_stats(profiler_output_path)

        except BaseException as e:

            profiler_output_path = self._get_path_of_component_of_trial(trial)
            profiler_output_path = os.path.join(profiler_output_path, "profiler.out")

            profiler.disable()
            profiler.dump_stats(profiler_output_path)

            raise e
        


        return to_return
            