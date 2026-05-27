
from automarl.component import Component, requires_input_process
from automarl.core.input_management import ParameterSignature
from automarl.components.loggers.result_logger import ResultLogger
from automarl.components.rl.evaluators.rl_component_evaluator import RLPipelineEvaluator
from automarl.components.rl.rl_pipeline import RLPipelineComponent

from sklearn.linear_model import LinearRegression


class RLLearningEvaluatorSlope(RLPipelineEvaluator):
    
    '''
    An evaluator specific for RL pipelines
    
    '''
    
    parameters_signature = {
        "number_of_episodes_percentage" : ParameterSignature(default_value=10),
        "init" : ParameterSignature(default_value=(0, "episode")),
        "final" : ParameterSignature(mandatory=False) 
    }
    

    def _process_input_internal(self):
        
        super()._process_input_internal()

        self.number_of_episodes_percentage = self.get_input_value("number_of_episodes_percentage")

        self.init = self.get_input_value(f"init")

        if not isinstance(self.init, (list, tuple)):
            self.init = (self.init, "episode")

        self.final = self.get_input_value(f"final")
        if self.final is not None and not isinstance(self.final, (list, tuple)):
            self.final = (self.final, "episode")

    
    def get_episode(self, number, results_logger : ResultLogger):

        df = results_logger.get_dataframe()

        if number is not None:

            (value, definition) = number

            if definition == "episode":
                return value

            elif definition == "step":
                # Find first episode where total_steps >= given step
                filtered = df[df["total_steps"] >= value]
                if filtered.empty:
                    return int(df["episode"].iloc[-1])
                return int(filtered.iloc[0]["episode"])

        else:
            return int(df["episode"].iloc[-1])


    # EVALUATION -------------------------------------------------------------------------------

    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), "result"]
    
    @requires_input_process
    def _evaluate(self, component_to_evaluate : RLPipelineComponent):
        super()._evaluate(component_to_evaluate)

        if isinstance(component_to_evaluate, ResultLogger):
            rl_trainer_logger = component_to_evaluate
        
        else:
            rl_trainer_logger : ResultLogger = component_to_evaluate.get_results_logger()

        init_episode = self.get_episode(self.init, rl_trainer_logger)
        final_episode = self.get_episode(self.init, rl_trainer_logger)

        number_of_episodes_to_test = max(2, (init_episode - final_episode) * (self.number_of_episodes_percentage / 100))

        init_episode = max(0, final_episode - number_of_episodes_to_test) 

        df = rl_trainer_logger.get_dataframe()

        if df.empty or len(df) < 2:
            return 0.0

        init_episode = self.get_episode(self.init, rl_trainer_logger)
        final_episode = self.get_episode(self.final, rl_trainer_logger)

        # Ensure correct ordering
        if final_episode < init_episode:
            init_episode, final_episode = final_episode, init_episode

        # Filter interval
        interval_df = df[
            (df["episode"] >= init_episode) &
            (df["episode"] <= final_episode)
        ]

        if len(interval_df) < 2:
            return 0.0

        # Determine number of episodes to use
        n_interval = len(interval_df)
        n_to_use = max(
            2,
            int(n_interval * (self.number_of_episodes_percentage / 100))
        )

        # Take last n_to_use episodes from interval
        selected_df = interval_df.tail(n_to_use)

        # Prepare regression
        X = selected_df["episode"].values.reshape(-1, 1)
        y = selected_df["episode_reward"].values

        model = LinearRegression()
        model.fit(X, y)

        slope = float(model.coef_[0])

        return {"result" : slope}