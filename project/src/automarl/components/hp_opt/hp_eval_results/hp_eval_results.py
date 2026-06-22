

import copy
import os
from automarl.consts import CONFIGURATION_FILE_NAME
from automarl.components.loggers.result_logger import ResultLogger, RESULTS_FILENAME
from automarl.components.hp_opt.hp_optimization_pipeline import HyperparameterOptimizationPipeline, OPTUNA_STUDY_PATH, BASE_CONFIGURATION_NAME
from automarl.components.rl.rl_pipeline import RLPipelineComponent
from automarl.serialization.json_component_utils import dict_from_path, gen_component_from
from automarl.utils.optuna_utils import load_study_from_database
import optuna
from optuna.importance import get_param_importances
from automarl.components.loggers.global_logger import globalWriteLine
from automarl.utils.files_utils import get_first_path_with_name
import pandas as pd


def filter_hp_opt_results_logger(optuna_study, results_logger, first_trial, last_trial):

    if first_trial > last_trial:
        raise ValueError(
            f"first_trial ({first_trial}) must be <= last_trial ({last_trial})"
        )

    original_df = results_logger.get_dataframe()

    filtered_df = original_df[
        (original_df["experiment"] >= first_trial)
        & (original_df["experiment"] <= last_trial)
    ].copy()

    range_suffix = f"{first_trial}-{last_trial}"

    new_results_filename = (
        f"{os.path.splitext(results_logger.results_filename)[0]}"
        f"_{range_suffix}.csv"
    )

    new_results_logger = ResultLogger(
        {
            "base_directory": results_logger.get_artifact_directory(),
            "artifact_relative_directory": "",
            "create_new_directory": False,
            "results_filename": new_results_filename,
            "results_columns": filtered_df.columns.tolist(),
        }
    )

    new_results_logger.process_input_if_not_processed()

    new_results_logger.dataframe = filtered_df
    new_results_logger.save_dataframe()

    study_name = (
        f"{optuna_study.study_name}_{range_suffix}"
        if optuna_study.study_name is not None
        else f"filtered_{range_suffix}"
    )

    new_optuna_study = optuna.create_study(
        direction=optuna_study.direction,
        study_name=study_name,
    )

    for trial in optuna_study.trials:
        if first_trial <= trial.number <= last_trial:
            new_optuna_study.add_trial(copy.deepcopy(trial))

    return new_optuna_study, new_results_logger



def get_hp_opt_results_logger(experiment_path, results_filename=RESULTS_FILENAME) -> ResultLogger:

    hp_results_columns = HyperparameterOptimizationPipeline.results_columns

    hyperparameter_optimization_results : ResultLogger = ResultLogger(input={
                                        "base_directory" : experiment_path,
                                        "artifact_relative_directory" : '',
                                        "results_filename" : RESULTS_FILENAME,
                                        "results_columns" : hp_results_columns,
                                        "create_new_directory" : False
                                      })

    hyperparameter_optimization_results.process_input_if_not_processed()

    return hyperparameter_optimization_results




def get_hp_opt_optuna_study(hp_results_logger : ResultLogger, database_path=OPTUNA_STUDY_PATH):
    optuna_study = load_study_from_database(database_path=hp_results_logger.get_artifact_directory() + '\\' + database_path)

    return optuna_study


def get_evaluation_statistics_per_step(hp_results_logger : ResultLogger):
    '''
    Returns, for each step, the average mean results of trials and average standard deviation of results of trials
    That is, the standard deviation and mean of results for the same experiment (different component indexes) and step
    '''

    df = hp_results_logger.get_dataframe()

    if df.empty:
        return {}

    # Step 1: stats per (experiment, step)
    exp_step_stats = (
        df.groupby(["experiment", "step"])["result"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Step 2: aggregate per step
    step_stats = (
        exp_step_stats.groupby("step")
        .agg(
            avg_mean=("mean", "mean"),
            avg_std=("std", "mean")
        )
        .reset_index()
    )

    # Convert to dictionary {step: {avg_mean, avg_std}}
    statistics_per_step = {
        int(row["step"]): {
            "avg_mean": float(row["avg_mean"]),
            "avg_std": float(row["avg_std"]) if not pd.isna(row["avg_std"]) else 0.0
        }
        for _, row in step_stats.iterrows()
    }

    return statistics_per_step



def get_component_stats_by_step(component_path):

    configuration_dict = dict_from_path(os.path.join(component_path, CONFIGURATION_FILE_NAME))

    raise NotImplementedError()


def print_optuna_trials_info(optuna_study):

    print("\n===== Trials Info =====")
    for t in optuna_study.trials:
        print(f"Trial {t.number}:")
        print(f"  State: {t.state}")
        print(f"  Value: {t.value}")
        print(f"  Params: {t.params}")
        print(f"  User attrs: {t.user_attrs}")
        print(f"  System attrs: {t.system_attrs}")
        print("-" * 40)



def print_optuna_param_importances(optuna_study):

    importances = get_param_importances(optuna_study)

    # Print nicely
    for param, importance in importances.items():
        print(f"{param}: {importance:.4f}")


#def get_params_in_optuna(optuna_study):
#
#    # DataFrame from study
#    df = optuna_study.trials_dataframe()
#    # Hyperparameter columns
#    param_cols = [c for c in df.columns if c.startswith("params_")]
#
#    return param_cols
#
#
#def plot_scattered_values_for_param(optuna_study, highlight_trials, param):
#
#    if highlight_trials is None:
#        highlight_trials = []
#
#
#def plot_scattered_values_for_params(optuna_study, highlight_trials, params=None):
#
#    if params is None:
#        return plot_scattered_values_for_all_params(optuna_study, highlight_trials)
#    
#    elif isinstance(params, list):
#        for param in params:
#            plot_scattered_values_for_param(optuna_study, highlight_trials, param)
#
#    else:
#        plot_scattered_values_for_param(optuna_study, highlight_trials, params)

def plot_scattered_values_for_all_params(optuna_study, highlight_trials=None):

    import optuna
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    if highlight_trials is None:
        highlight_trials = []

    # DataFrame from study
    df = optuna_study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]

    # Hyperparameter columns
    param_cols = [c for c in df.columns if c.startswith("params_")]

    trial_numbers = df["number"].values

    # Normalize trial numbers to [0,1]
    min_trial = trial_numbers.min()
    max_trial = trial_numbers.max()
    
    norm_trials =  ( (trial_numbers - min_trial) / (max_trial - min_trial + 1e-8) )
    highlight_norm_trials = ( (trial_numbers - min_trial) / (max_trial - min_trial + 1e-8) ) 

    # Invert so earlier trials are more transparent
    alphas = 0.1 + 0.9 * norm_trials  # alpha between 0.1 and 1.0


    # Plot each parameter
    for param in param_cols:
        plt.figure(figsize=(7, 5))

        x = df[param].values
        y = df["value"].values

        zipped_all_trials = iter(zip(x, y, alphas))

        xi, yi, ai = next(zipped_all_trials)
        plt.scatter(xi, yi, alpha=ai, color="red", label="All trials")

        # Base plot: all trials
        for xi, yi, ai in zipped_all_trials:
            plt.scatter(xi, yi, alpha=ai, color="red")
        

        base_name = param.replace("params_", "")

        # Collect all highlighted X,Y
        hl_x = []
        hl_y = []
        for t in highlight_trials:
            if base_name in t.params:
                hl_x.append(t.params[base_name])
                hl_y.append(t.value)

        # Scatter highlighted points (if exist)
        if len(hl_x) > 0:
            plt.scatter(
                hl_x, hl_y,
                color="blue",
                s=70,
                label="Highlighted trials"
            )

        plt.xlabel(base_name)
        plt.ylabel("Objective")
        plt.title(f"{base_name} vs Objective (highlighted trials as a group)")
        plt.legend()
        plt.grid(True)
        plt.show()





#def partial dependency:

#   import pandas as pd
#   
#   # Convert trials into a DataFrame
#   df = optuna_study.trials_dataframe(attrs=("number", "value", "params", "state"))
#   
#   # Keep only completed trials
#   df = df[df["state"] == "COMPLETE"]
#   
#   X = df[[c for c in df.columns if c.startswith("params_")]]
#   y = df["value"]
#   
#   from sklearn.ensemble import RandomForestRegressor
#   
#   # You could also use GradientBoosting, XGBoost, LightGBM, etc.
#   surrogate = RandomForestRegressor(n_estimators=200, random_state=0)
#   surrogate.fit(X, y)
#   
#   from sklearn.inspection import PartialDependenceDisplay
#   
#   # Single parameter PDP
#   PartialDependenceDisplay.from_estimator(surrogate, X, features=["params_learning_rate"])
#   
#   # 2D PDP for interaction between two hyperparams
#   PartialDependenceDisplay.from_estimator(
#       surrogate, X, features=[("params_learning_rate", "params_optimization_interval")]
#   )



def study_of_agent_trainer(agent_trainer_name : str, results_logger : ResultLogger,
                           #x_axis_to_use='episode',
                           x_axis_to_use='episode',
                           y_axis_to_use='episode_reward',
                           aggregate_number=10,
                           prefix=''
                            ):


    #results_logger.plot_graph(x_axis='episode', y_axis=[('total_reward', name)], to_show=False)

    results_logger.plot_confidence_interval(x_axis=x_axis_to_use, 
                                            y_column=y_axis_to_use,
                                            show_std=True, 
                                            to_show=False, 
                                            y_values_label=f"mov_avg_std ({aggregate_number})", 
                                            aggregate_number=aggregate_number)

    #results_logger.plot_linear_regression(x_axis='episode', y_axis='episode_reward', to_show=False)

    #results_logger.plot_piecewise_linear_regression(x_axis='episode', y_axis='episode_reward', to_show=False, n_segments=10)

    results_logger.plot_polynomial_regression(x_axis=x_axis_to_use, y_axis=y_axis_to_use, to_show=False, degrees=4)

    results_logger.plot_current_graph(title=f"{prefix}{agent_trainer_name}", y_label=y_axis_to_use)


def study_of_agent_trainers(path_where_agent_trainer_exist : str, prefix = ''):


    for path_of_child in os.listdir(path_where_agent_trainer_exist):

        if path_of_child.endswith("_trainer"):

            trainer_name = os.path.basename(path_of_child)
            results_logger = ResultLogger({
                "create_new_directory" : False,
                "base_directory" : os.path.join(path_where_agent_trainer_exist, path_of_child),
                "artifact_relative_directory" : ''
            })

            study_of_agent_trainer(trainer_name, results_logger, prefix=prefix)


def study_of_configuration(configuration_name : str, results_logger : ResultLogger,
                           #x_axis_to_use='episode',
                           x_axis_to_use='total_steps',
                           y_axis_to_use='episode_reward',
                           aggregate_number=10,
                                    study_evaluations=True,
                            ):


    #results_logger.plot_graph(x_axis='episode', y_axis=[('total_reward', name)], to_show=False)

    results_logger.plot_confidence_interval(x_axis=x_axis_to_use, 
                                            y_column=y_axis_to_use,
                                            show_std=True, 
                                            to_show=False, 
                                            y_values_label=f"mov_avg_std ({aggregate_number})", 
                                            aggregate_number=aggregate_number)

    #results_logger.plot_linear_regression(x_axis='episode', y_axis='episode_reward', to_show=False)

    #results_logger.plot_piecewise_linear_regression(x_axis='episode', y_axis='episode_reward', to_show=False, n_segments=10)

    results_logger.plot_polynomial_regression(x_axis=x_axis_to_use, y_axis=y_axis_to_use, to_show=False, degrees=4)

    results_logger.plot_current_graph(title=configuration_name, y_label=y_axis_to_use)

    if study_evaluations:

        evaluations_logger = get_evaluations_results_logger(os.path.join(results_logger.get_artifact_directory(), ".."))

        if evaluations_logger != None:
            study_of_evaluations(configuration_name, evaluations_logger)

        else:
            print(f"Study has no evaluations to study")


def study_of_components_for_configuration(configuration_name : str, results_loggers : dict[str, ResultLogger],
                           #x_axis_to_use='episode',
                           x_axis_to_use='total_steps',
                           y_axis_to_use='episode_reward',
                           aggregate_number=10,
                            colors_for_component_indexes : dict = None
                            ):

    
    for component_name, results_logger in results_loggers.items():

            color = colors_for_component_indexes[int(component_name)] if colors_for_component_indexes is not None else None

            results_logger.plot_confidence_interval(x_axis=x_axis_to_use, 
                                            y_column=y_axis_to_use,
                                            show_std=True, 
                                            to_show=False, 
                                            y_values_label=f"mov_avg_std_{component_name} ({aggregate_number})", 
                                            aggregate_number=aggregate_number,
                                            color=color,
                                            alpha=0.1)   



    results_logger.plot_current_graph(title=configuration_name, y_label=y_axis_to_use)     



def get_evaluations_path(base_path):

    return get_first_path_with_name(base_path, "evaluations")


def get_evaluations_results_logger(base_path):

    evaluations_path = get_evaluations_path(base_path)

    if evaluations_path == None:
        return None
    

    return ResultLogger(input={
                                                "base_directory" : evaluations_path,
                                                "artifact_relative_directory" : '',
                                                "create_new_directory" : False,
                                                "results_filename" : "evaluations.csv"
                                              })




def study_of_evaluations(configuration_name : str, results_logger : ResultLogger,
                           #x_axis_to_use='episode',
                           x_axis_to_use='evaluation',
                           y_axis_to_use='result'
                           ):
    
    results_logger.plot_bar_graph(x_axis=x_axis_to_use, y_axis=y_axis_to_use, to_show=False)

    results_logger.plot_linear_regression(x_axis='evaluation', y_axis='result', to_show=False, color="orange")


    results_logger.plot_current_graph(title=f"{configuration_name}_evaluations", y_label=y_axis_to_use, y_min=0)


def get_results_of_configuration_in_path(configuration_path, 
                                         configurations_results_relative_path,
                                         results_path=RESULTS_FILENAME) -> ResultLogger:

        results_logger_of_config = ResultLogger(input={
                                    "results_filename" : results_path,
                                    "base_directory" : os.path.join(configuration_path, configurations_results_relative_path),
                                    "artifact_relative_directory" : '',
                                    "create_new_directory" : False

                                  })

        results_logger_of_config.process_input()

        return results_logger_of_config



def get_results_of_configurations(experiment_path,
                                   base_configuration_name=BASE_CONFIGURATION_NAME,
                                  results_path=RESULTS_FILENAME) -> dict[str, ResultLogger]:
    
    '''Gets directly the results of configurations in a path'''

    results_of_configurations : dict[str, ResultLogger] = {}

    configurations_results_relative_path = "RLTrainerComponent"

    for configuration_name in os.listdir(experiment_path):

        if configuration_name.startswith(base_configuration_name):

            configuration_path = os.path.join(experiment_path, configuration_name)

            if os.path.isdir(configuration_path):  # Ensure it's a file, not a subdirectory

                try:
                    results_of_configurations[configuration_name] = get_results_of_configuration_in_path(
                        configuration_path,
                        configurations_results_relative_path,
                        results_path
                    )

                except Exception as e:
                    print(f"Did not manage to get configuration {configuration_name} due to error {e}")

            else:
                print(f"WARNING: Configuration path with name {configuration_name} is not a directory")

            
    return results_of_configurations


def get_results_of_configurations_components(experiment_path,
                                   base_configuration_name=BASE_CONFIGURATION_NAME,
                                   configurations_results_relative_path = "RLTrainerComponent",
                                  results_path=RESULTS_FILENAME,
                                  use_multiple=True) -> dict[str, dict[str, ResultLogger]]:

    results_of_configurations : dict[str, dict[str, ResultLogger]] = {}

    for configuration_name in os.listdir(experiment_path):

        if configuration_name.startswith(base_configuration_name):

            configuration_path = os.path.join(experiment_path, configuration_name)

            if os.path.isdir(configuration_path):  # Ensure it's a file, not a subdirectory

                configuration_dict = {}

                if use_multiple:

                    for configuration_component_name in os.listdir(configuration_path):

                        configuration_component_path = os.path.join(configuration_path, configuration_component_name)

                        try:
                            configuration_dict[configuration_component_name] = get_results_of_configuration_in_path(
                                configuration_component_path,
                                configurations_results_relative_path,
                                results_path
                            )

                        except Exception as e:
                            print(f"Did not manage to get configuration {configuration_name} due to error {e}")
                
                else:
                    configuration_dict["0"] = get_results_of_configuration_in_path(
                                configuration_path,
                                configurations_results_relative_path,
                                results_path
                            )

                results_of_configurations[configuration_name] = configuration_dict

            else:
                globalWriteLine(f"WARNING: Configuration path with name {configuration_name} is not a directory")

            
    return results_of_configurations


def get_pruned_trials(optuna_study):

    '''Gets pruned trials per steps, organized by steps'''

    pruned_optuna_trials = [trial for trial in optuna_study.trials if trial.state == optuna.trial.TrialState.PRUNED]

    pruned_optuna_trials_per_steps : dict[int, list[optuna.trial.FrozenTrial]] = {} #the pruned trials by the number of completed steps

    for pruned_optuna_trial in pruned_optuna_trials:

        n_completed_steps = len(pruned_optuna_trial.intermediate_values)

        try:
            list_of_pruned = pruned_optuna_trials_per_steps[n_completed_steps]

        except:
            list_of_pruned = []
            pruned_optuna_trials_per_steps[n_completed_steps] = list_of_pruned    

        list_of_pruned.append(pruned_optuna_trial)


    for list_of_pruned in pruned_optuna_trials_per_steps.values():
        list_of_pruned.sort(key=lambda trial: trial.value) 



    pruned_trials = [f'{BASE_CONFIGURATION_NAME}_{trial.number + 1}' for trial in optuna_study.trials if trial.state == optuna.trial.TrialState.PRUNED]

    return pruned_optuna_trials, pruned_optuna_trials_per_steps, pruned_trials


def get_trials_with_decreasing_intermediates(study):

    bad_trials = []

    for trial in study.trials:
        ivals = trial.intermediate_values

        # Need at least 2 intermediate values to compare
        if len(ivals) < 2:
            continue

        # Sort by step index (intermediate_values is a dict: step -> value)
        steps = sorted(ivals.keys())
        values = [ivals[s] for s in steps]

        # Check if any later value is lower than any earlier value
        ever_decreased = any(values[j] < values[i] 
                             for i in range(len(values)) 
                             for j in range(i + 1, len(values)))

        if ever_decreased:
            bad_trials.append(trial)

    return bad_trials

def print_intermidiate_values(trial_list : optuna.trial.FrozenTrial):

    import optuna

    if len(trial_list) == 0:
        return
    
    for trial in trial_list:

        # Sort the intermediate values by step index
        steps = sorted(trial.intermediate_values.keys())
        values = [trial.intermediate_values[s] for s in steps]

        # Format as: trial X: v1, v2, v3
        values_str = ", ".join(str(v) for v in values)
        print(f"trial {trial.number}: {values_str}")



def plot_bayesian_optimization_for_param(
    optuna_study,
    trials_to_use,
    hyperparam,
    acquisition_function="ei",
    xi=0.01,
    n_points=300,
    show_confidence=True,
    use_log_x=False,
    output_dir=None,
    xlim=None,
    ylim=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import optuna

    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

    if trials_to_use is None:
        trials_to_use = optuna_study.trials

    if trials_to_use is None or len(trials_to_use) == 0:
        raise ValueError("trials_to_use must contain at least one trial.")

    def expected_improvement(mu, sigma, best_y, minimize=True, xi=0.0):
        sigma = np.maximum(sigma, 1e-12)

        if minimize:
            improvement = best_y - mu - xi
        else:
            improvement = mu - best_y - xi

        z = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma <= 1e-12] = 0.0
        return ei

    def upper_confidence_bound(mu, sigma, minimize=True, xi=2.0):
        if minimize:
            # maximize acquisition
            return -(mu - xi * sigma)
        else:
            return mu + xi * sigma

    # Filter trials actually used
    filtered_trials = []
    plot_x = []
    plot_y = []
    plot_trial_numbers = []

    for t in trials_to_use:
        if (
            t.state == optuna.trial.TrialState.COMPLETE
            and hyperparam in t.params
            and t.value is not None
        ):
            try:
                x_val = float(t.params[hyperparam])
                y_val = float(t.value)

                if use_log_x:
                    if x_val <= 0:
                        continue
                    x_val = np.log10(x_val)

                filtered_trials.append(t)
                plot_x.append(x_val)
                plot_y.append(y_val)
                plot_trial_numbers.append(t.number)
            except Exception:
                pass

    if len(plot_x) < 2:
        raise ValueError(
            f"Need at least 2 COMPLETE trials with param '{hyperparam}' in trials_to_use."
        )

    plot_x = np.array(plot_x, dtype=float)
    plot_y = np.array(plot_y, dtype=float)
    plot_trial_numbers = np.array(plot_trial_numbers, dtype=float)

    # Sort by x for cleaner visualization
    order = np.argsort(plot_x)
    plot_x = plot_x[order]
    plot_y = plot_y[order]
    plot_trial_numbers = plot_trial_numbers[order]

    # Alpha based on trial number among filtered trials
    min_trial = plot_trial_numbers.min()
    max_trial = plot_trial_numbers.max()
    alphas = 0.15 + 0.85 * (
        (plot_trial_numbers - min_trial) / (max_trial - min_trial + 1e-8)
    )

    data_x_min = np.min(plot_x)
    data_x_max = np.max(plot_x)
    
    if data_x_min == data_x_max:
        padding = 1.0 if data_x_min == 0 else abs(data_x_min) * 0.1
        data_x_min -= padding
        data_x_max += padding
    else:
        padding = 0.05 * (data_x_max - data_x_min)
        data_x_min -= padding
        data_x_max += padding
    
    # Use plotting limits for GP evaluation if provided
    if xlim is not None:
        x_min, x_max = xlim
    else:
        x_min, x_max = data_x_min, data_x_max
    
    x_grid = np.linspace(x_min, x_max, n_points)

    # Standardize X and y for GP stability
    x_mean = plot_x.mean()
    x_std = plot_x.std() + 1e-12
    y_mean = plot_y.mean()
    y_std = plot_y.std() + 1e-12

    X_train = ((plot_x - x_mean) / x_std).reshape(-1, 1)
    X_grid = ((x_grid - x_mean) / x_std).reshape(-1, 1)
    y_train = (plot_y - y_mean) / y_std

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=False,  # already standardized
        n_restarts_optimizer=10,
        random_state=0,
    )
    gp.fit(X_train, y_train)

    mu_std, sigma_std = gp.predict(X_grid, return_std=True)

    # Transform GP prediction back to original y scale
    mu = mu_std * y_std + y_mean
    sigma = sigma_std * y_std

    minimize = optuna_study.direction == optuna.study.StudyDirection.MINIMIZE
    best_y = np.min(plot_y) if minimize else np.max(plot_y)

    acquisition_function = acquisition_function.lower()
    
    if acquisition_function != "none":
    
        if acquisition_function == "ei":
            acq = expected_improvement(mu, sigma, best_y, minimize=minimize, xi=xi)
            acq_label = "Expected Improvement"
        elif acquisition_function == "ucb":
            acq = upper_confidence_bound(mu, sigma, minimize=minimize, xi=max(xi, 1.0))
            acq_label = "Confidence Bound"

        else:
            raise ValueError(
                f"Unsupported acquisition_function '{acquisition_function}'. Use 'ei' or 'ucb'."
            )
        
    else:
        acq = None
        acq_label = None

    fig, ax1 = plt.subplots(figsize=(9, 6))

    first = True
    for x_i, y_i, a_i in zip(plot_x, plot_y, alphas):
        ax1.scatter(
            x_i,
            y_i,
            alpha=a_i,
            color="red",
            label=f"Trials used ({len(filtered_trials)})" if first else None,
        )
        first = False

    ax1.plot(x_grid, mu, color="green", linewidth=2, label="GP mean")

    if show_confidence:
        ax1.fill_between(
            x_grid,
            mu - 1.96 * sigma,
            mu + 1.96 * sigma,
            color="green",
            alpha=0.2,
            label="GP 95% CI",
        )

    ax1.set_xlabel(f"log10({hyperparam})" if use_log_x else hyperparam)
    ax1.set_ylabel("Objective")

    if xlim is not None:
        ax1.set_xlim(*xlim)

    if ylim is not None:
        ax1.set_ylim(*ylim)

    if acq is not None:

        ax2 = ax1.twinx()
        ax2.plot(
            x_grid,
            acq,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=acq_label,
        )
        ax2.set_ylabel("Acquisition")

        best_acq_idx = int(np.argmax(acq))
        ax2.scatter(
            [x_grid[best_acq_idx]],
            [acq[best_acq_idx]],
            color="purple",
            s=80,
            label="Best acquisition point",
        )
    else:
        ax2 = None

    ax1.set_title(f"{hyperparam} vs Objective with GP")
    ax1.grid(True)
    lines1, labels1 = ax1.get_legend_handles_labels()

    if ax2 is not None:
        lines2, labels2 = ax2.get_legend_handles_labels()    
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    
    else:
        ax1.legend(lines1, labels1, loc="best")

    plt.tight_layout()

    if output_dir is not None:
        import os

        filename = f"{optuna_study.study_name}_{hyperparam}_bayesian_fit.svg"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(
            filepath,
            format="svg",
            bbox_inches="tight",
        )

    plt.show()

    print("Learned kernel:", gp.kernel_)

def plot_bayesian_optimization_for_params(
    optuna_study,
    trials_to_use=None,
    params=None,
    acquisition_function="ei",
    use_log_x=False,
    acquisition_points=300,
    output_dir=None,
    x_lim=None,
    y_lim=None
):
    
    if trials_to_use is None:
        trials_to_use = optuna_study.trials

    if trials_to_use is None or len(trials_to_use) == 0:
        raise ValueError("trials_to_use must contain at least one trial.")

    if params is None:
        params = sorted({
            param_name
            for t in trials_to_use
            if t.state == optuna.trial.TrialState.COMPLETE
            for param_name in t.params.keys()
        })

    if isinstance(params, list):
        for param in params:
            plot_bayesian_optimization_for_param(
                optuna_study=optuna_study,
                trials_to_use=trials_to_use,
                hyperparam=param,
                acquisition_function=acquisition_function,
                use_log_x=use_log_x,
                n_points=acquisition_points,
                output_dir=output_dir,
                xlim=x_lim,
                ylim=y_lim,
            )
    else:
        plot_bayesian_optimization_for_param(
            optuna_study=optuna_study,
            trials_to_use=trials_to_use,
            hyperparam=params,
            acquisition_function=acquisition_function,
            use_log_x=use_log_x,
            n_points=acquisition_points,
            output_dir=output_dir,
                xlim=x_lim,
                ylim=y_lim,
        )