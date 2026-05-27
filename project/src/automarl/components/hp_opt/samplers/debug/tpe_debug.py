from automarl.components.hp_opt.samplers.sampler import OptunaSamplerWrapper
from automarl.components.loggers.debug.component_with_logging_debug import ComponentDebug
import optuna

class OptunaSamplerWrapperDebug(OptunaSamplerWrapper):

    is_debug_schema = True

    def process_input(self):
        super().process_input()

    def _patch_sampler(self):

        sampler = self.optuna_sampler

        if sampler is None:
            raise Exception("Sampler not initialized")

        # Store original methods
        sampler._original_sample_relative = sampler.sample_relative
        sampler._original_sample_independent = sampler.sample_independent

        def debug_sample_relative(study, trial, search_space):

            self.lg.writeLine("Optuna sampler: sample_relative called")
            self.lg.writeLine(f"Search space: {list(search_space.keys())}")

            result = sampler._original_sample_relative(study, trial, search_space)

            self.lg.writeLine(f"Relative sample result: {result}")

            # If this is TPE with grouping we can inspect internal groups
            if isinstance(sampler, optuna.samplers.TPESampler):

                group_obj = getattr(sampler, "_search_space_group", None)

                if group_obj is not None:

                    groups = getattr(group_obj, "_groups", None)

                    if groups is not None:
                        formatted = [list(g) for g in groups]
                        self.lg.writeLine(f"TPE parameter groups: {formatted}")

            return result

        def debug_sample_independent(study, trial, param_name, param_distribution):

            self.lg.writeLine("Optuna sampler: sample_independent called")
            self.lg.writeLine(f"Parameter: {param_name}")
            self.lg.writeLine(f"Distribution: {param_distribution}")

            result = sampler._original_sample_independent(
                study, trial, param_name, param_distribution
            )

            self.lg.writeLine(f"Independent sample result: {param_name}={result}")

            return result

        # Monkey patch methods
        sampler.sample_relative = debug_sample_relative
        sampler.sample_independent = debug_sample_independent