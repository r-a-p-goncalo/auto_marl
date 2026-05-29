from automarl.components.basic_components.dynamic_value import get_value_or_dynamic_value
from automarl.component import Component, ParameterSignature, requires_input_process

from automarl.core.advanced_input_management import ComponentParameterSignature
from automarl.components.fundamentals.acessories import AcessoryComponent


from automarl.components.loggers.logger_component import ComponentWithLogging
from automarl.components.loggers.result_logger import ResultLogger
from automarl.components.rl.learners.learner_component import LearnerSchema
from automarl.components.rl.trainers.agent_trainer.agent_trainer_component import AgentTrainer
import torch


class AgentTrainerTrainingEnder(AcessoryComponent, ComponentWithLogging):
    
    '''
    Detects convergence by comparing old and new values and noting their differences
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                        "check_interval" : ParameterSignature(default_value=10)
                        }    
    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()
                
        self.affected_component : AgentTrainer = self.affected_component

        if self.affected_component is None:
            raise Exception(f"Needs training")
        
        self.affected_component.initialize_external_end_request(self.name)

        self.check_interval = self.get_input_value("check_interval")

        self.current_check_counter = 0


    def check_if_should_end(self, values):
        return False

    @requires_input_process
    def as_fun(self, values : dict = None):
        '''To be called after a functionality is executed'''

        if self.current_check_counter == 0:
            if self.check_if_should_end(values):
                self.affected_component.request_end_from_external(self.name)
            
            else:
                self.affected_component.request_continue_from_external(self.name)
        
        self.current_check_counter += 1

        if self.current_check_counter == self.check_interval:
            self.current_check_counter = 0




class AgentTrainerConvergenceDetector(AgentTrainerTrainingEnder):
    
    '''
    Detects convergence by comparing old and new values and noting their differences
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                                                       
                        "standard_deviation_treshold" : ParameterSignature(default_value=10),

                        "min_avg_value" : ParameterSignature(mandatory=False),
                        
                        "value_key" : ParameterSignature(default_value="episode_reward"),

                        "n_values_to_use" : ParameterSignature(default_value=100),

                        }    
    
    
    def _process_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._process_input_internal()

        self.standard_deviation_treshold = self.get_input_value("standard_deviation_treshold")

        self.value_key = self.get_input_value("value_key")
        self.n_values_to_use = self.get_input_value("n_values_to_use")
        self.min_avg_value = self.get_input_value("min_avg_value")

        self.lg.writeLine(f"Convergence will be noted with standard deviation, using an average of {self.n_values_to_use} values, that must be bellow {self.standard_deviation_treshold}")



    def check_if_should_end(self, values):
        """
        Checks if the standard deviation of the last n_values_to_use
        values of value_key is below the defined threshold.
        """

        # Access trainer results logger
        results_logger : ResultLogger = self.affected_component.get_results_logger()

        # Get dataframe (assumes ResultLogger exposes dataframe attribute)
        df = results_logger.get_dataframe()

        if df is None or len(df) < self.n_values_to_use:
            return False

        if self.value_key not in df.columns:
            raise Exception(
                f"Value key '{self.value_key}' not found in results columns: {df.columns}"
            )

        # Get last N values
        last_values = df[self.value_key].tail(self.n_values_to_use).values
        last_values = torch.tensor(last_values, dtype=torch.float32)

        std = float(last_values.std().item())
        to_return = std < self.standard_deviation_treshold

        if self.min_avg_value is not None:
            avg = float(last_values.mean().item())
            to_return = to_return and avg >= self.min_avg_value

        return to_return
    




class AgentTrainerSlopeConvergenceDetector(AgentTrainerTrainingEnder):
    
    '''
    Detects convergence by computing the slope of the last n episode values.
    If slope is below a given threshold, convergence is assumed.
    '''

    # PARAMETERS --------------------------------------------------------------------------

    parameters_signature = {
                        
        "slope_threshold": ParameterSignature(default_value=0.1),
        
        "value_key": ParameterSignature(default_value="episode_reward"),

        "n_values_to_use": ParameterSignature(default_value=100),

    }

    # INITIALIZATION ----------------------------------------------------------------------

    def _process_input_internal(self):
        
        super()._process_input_internal()


        self.slope_threshold = self.get_input_value("slope_threshold")
        self.value_key = self.get_input_value("value_key")
        self.n_values_to_use = self.get_input_value("n_values_to_use")

        self.lg.writeLine(
            f"Slope convergence detector initialized. "
            f"Using last {self.n_values_to_use} values of '{self.value_key}'. "
            f"Converged if slope < {self.slope_threshold}"
        )

    # CORE LOGIC --------------------------------------------------------------------------

    def _compute_slope(self, y_values: torch.Tensor) -> float:
        """
        Computes linear regression slope using least squares:
        slope = cov(x,y) / var(x)
        """

        n = len(y_values)
        x = torch.arange(n, dtype=torch.float32)

        x_mean = x.mean()
        y_mean = y_values.mean()

        cov_xy = ((x - x_mean) * (y_values - y_mean)).sum()
        var_x = ((x - x_mean) ** 2).sum()

        if var_x == 0:
            return 0.0

        slope = cov_xy / var_x
        return slope.item()

    def check_if_should_end(self, values):

        results_logger: ResultLogger = self.affected_component.get_results_logger()
        df = results_logger.get_dataframe()

        if df is None or len(df) < self.n_values_to_use:
            return False

        if self.value_key not in df.columns:
            raise Exception(
                f"Value key '{self.value_key}' not found in results columns: {df.columns}"
            )

        last_values = df[self.value_key].tail(self.n_values_to_use).values
        y_tensor = torch.tensor(last_values, dtype=torch.float32)

        slope = self._compute_slope(y_tensor)

        return slope < self.slope_threshold


    