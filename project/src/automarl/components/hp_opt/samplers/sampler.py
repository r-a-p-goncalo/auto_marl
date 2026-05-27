

from automarl.components.basic_components.seeded_component import SeededComponent
from automarl.component import requires_input_process
from automarlcore.input_management import ParameterSignature
from automarl.components.loggers.logger_component import ComponentWithLogging
import optuna

class OptunaSamplerComponent(ComponentWithLogging):

    def __int__(self):
        pass

    @requires_input_process
    def get_optuna_sampler(self) ->  optuna.samplers.BaseSampler:
        '''Returns a sampler optuna can use, the one that is meant to be passed to the study'''
        return self
    

class OptunaSamplerWrapper(OptunaSamplerComponent, SeededComponent):

    parameters_signature = {
        "optuna_sampler" : ParameterSignature(default_value="TreeParzen"),
        "sampler_input" : ParameterSignature(mandatory=False)
    }

    def process_input(self):

        super().process_input()

        self._initialize_sampler()


    def _initialize_sampler(self):

        self.optuna_sampler : optuna.samplers.BaseSampler = self.get_input_value("optuna_sampler")

        if isinstance(self.optuna_sampler, str):
            self._initialize_sampler_from_str(self.optuna_sampler)

        elif isinstance(self.optuna_sampler, type):
            self._initialize_sampler_from_class(self.optuna_sampler)

        else:
            raise Exception("Non valid type for sampler")
        

    def _initialize_sampler_from_class(self):

        raise NotImplementedError()
        
        
    
    def _initialize_sampler_from_str(self, sampler_str):

        self.lg.writeLine(f"Initializing sampler with string {sampler_str}")

        sampler_input = self.get_input_value("sampler_input")
        sampler_input = {} if sampler_input is None else sampler_input

        if sampler_input is not None:
            self.lg.writeLine(f"Sampler input: {sampler_input}")
        
        if sampler_str == "TreeParzen":
            
            self.optuna_sampler : optuna.samplers.BaseSampler = optuna.samplers.TPESampler(seed=self.seed, **sampler_input)

        elif sampler_str == "Random":
            self.optuna_sampler : optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=self.seed, **sampler_input)

        
        else:
            raise NotImplementedError(f"Non valid string for sampler '{sampler_str}'") 
        
        
    @requires_input_process
    def get_optuna_sampler(self) -> optuna.samplers.BaseSampler:
        '''Returns a sampler optuna can use'''
        return self.optuna_sampler