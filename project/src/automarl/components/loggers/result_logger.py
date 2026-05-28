import os
import pandas as pd
from automarl.component import Component, ParameterSignature, requires_input_process
from automarl.components.loggers.logger_component import LoggerSchema
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import pandas
from typing import Dict

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import numpy as np
from automarl.utils.statistics.statistics import aggregate_series

RESULTS_FILENAME = 'results.csv'

class ResultLogger(LoggerSchema):

    # TODO: empty parameters_signature should be able to be removed
    # TODO: verify order at which keys are verified
    parameters_signature = {
            "results_filename" : ParameterSignature(default_value=RESULTS_FILENAME, description="The filename of the results file, in this case a csv file"),
            "results_columns" : ParameterSignature(possible_types=[list], description="The columns (metrics) of the results", mandatory=False),
            "save_results_on_log" : ParameterSignature(default_value=False, description="If the results should be save on the disk after each log")
        } 
    

    # INITIALIZATION --------------------------------------------------------

    def _process_input_internal(self): #this is the best method to have initialization done right after
        
        super()._process_input_internal()
        
        self.results_filename = self.get_input_value("results_filename")
        
        self._initialize_dataframe()
        
        self.save_on_log = self.get_input_value("save_results_on_log")
    
    
    def _initialize_dataframe(self):
        
        try:
        
            dataframe_on_folder = self.loadDataframe(filename=self.results_filename)

            self.dataframe = dataframe_on_folder
            self.columns = self.dataframe.columns
        
        except Exception as e:
                        
            self.columns = self.get_input_value("results_columns")

            if self.columns == None:
                raise Exception(f"Results logger dataframe did not exist in artifact directory {self.get_artifact_directory()} with filename {self.results_filename} and no columns passed")

            self.dataframe = pandas.DataFrame(columns=self.columns)
            self._save_dataframe()
            
        


                
        
    # USAGE -------------------------------------------------------------------
 
 
    @requires_input_process           
    def log_results(self, results : Dict[str, list]):
                
        for key, value in results.items(): # TODO: WE SHOULD JUST MAKE results = [results] and it would work
            if not isinstance(value, list):
                results[key] = [results[key]]
                
        results_df = pandas.DataFrame(results, columns=self.columns)
        
        self.dataframe = pandas.concat((self.dataframe, results_df), ignore_index=True) 
        
        if self.save_on_log:
            self._save_dataframe()
              
        
    
    @requires_input_process    
    def save_dataframe(self):
        self._save_dataframe()
                
    def _save_dataframe(self):
        
        '''For internal use, does not trigger input processing'''
        
        self.saveDataframe(self.dataframe, filename=self.results_filename)
        
    
    @requires_input_process
    def get_results_columns(self):
        return self.columns
        

   # RETURN RESULTS ------------------------------------------------------------------------------------------------
    
    @requires_input_process
    def get_number_of_rows(self):
        
        return len(self.dataframe)
        
    @requires_input_process
    def get_last_results(self):
                        
        return self.dataframe.tail(1).to_dict(orient="records")[0]
    
    @requires_input_process
    def get_n_last_results(self, n_results):
        
        return self.dataframe.tail(n_results).to_dict(orient="records")[0]
    
    @requires_input_process
    def get_avg_n_last_results(self, n_results, column):
        
        return self.dataframe[column].tail(n_results).mean()
    
    
    @requires_input_process
    def get_std_n_last_results(self, n_results, column):
        
        return self.dataframe[column].tail(n_results).std()
    
    @requires_input_process
    def get_avg_and_std_n_last_results(self, n_results, column):
        
        return self.get_avg_n_last_results(n_results, column), self.get_std_n_last_results(n_results, column)
    
    @requires_input_process
    def get_sorted_dataframe(self, column, ascending):
        
        return self.dataframe.sort_values(by=column, ascending=ascending)
    
    @requires_input_process
    def get_n_last_ordered_results(self, n, column, ascending = True):
        
        '''Returns the best n results'''
        
        ordered_dataframe = self.get_sorted_dataframe(column=column, ascending=ascending)
        
        return ordered_dataframe.tail(n).to_records()
    
    @requires_input_process
    def get_dataframe(self) -> pandas.DataFrame:
        return self.dataframe

    def get_grouped_dataframes(self, key_to_group_by) -> Dict[str, pandas.DataFrame]:
        
        grouped_dataframes = self.dataframe.groupby(key_to_group_by)

        grouped_dataframes_dict = {k: v for k, v in grouped_dataframes}
        
        return grouped_dataframes_dict

    # GRAPHS ------------------------------------------------------------------------------------------------------------------

    def generate_values_from_dataframe(self, x_axis, y_axis, df : pandas.DataFrame = None):

        if df is None:
            df = self.get_dataframe()

        if isinstance(x_axis, str):
            x_cols = [x_axis]
        else:
            x_cols = list(x_axis)

        for col in x_cols + [y_axis]:
            if col not in self.dataframe.columns:
                raise KeyError(
                    f"Column '{col}' not found in dataframe. "
                    f"Available columns: {self.dataframe.columns}"
                )

        if len(x_cols) == 1:
            x_values = df[x_cols[0]]
            x_label_name = x_cols[0]
        else:
            x_values = df[x_cols].astype(str).agg(" | ".join, axis=1)
            x_label_name = " & ".join(x_cols)

        y_values = df[y_axis]
        y_label_name = y_axis

        return x_values, x_label_name, y_values, y_label_name
    

    def filter_dataframe(self, fixed_value_tuple = [], df : pandas.DataFrame = None) -> pandas.DataFrame:

        df = self.dataframe if df is None else df

        if fixed_value_tuple is not None:

            if not isinstance(fixed_value_tuple, list):
                fixed_value_tuple = [fixed_value_tuple]
            
            for (column_name, fixed_value) in fixed_value_tuple:
            
                df = df[df[column_name] == fixed_value]

        return df

    @requires_input_process
    def plot_bar_graph(self, x_axis : str, y_axis : str, title : str = '', save_path: str = None, to_show=True, y_label='', lim_y=True, fixed_value_tuple=None, color = None, group_by=None):
        
        """
        Plots a graph using the dataframe stored in ResultLogger.
        """
        
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")
        
        df = self.filter_dataframe(fixed_value_tuple)

        if group_by is not None:
            df = df.sort_values(by=group_by)

        x_values, x_label_name, y_values, y_label_name = self.generate_values_from_dataframe(x_axis, y_axis, df)

        if len(x_values) == 0:
            return

        plt.bar(x_values, y_values, color=color)

        plt.xlabel(x_label_name)
        plt.ylabel(y_label_name)

        if lim_y:
            all_values = self.dataframe[y_axis].values.flatten()
            y_min, y_max = all_values.min(), all_values.max()

            if y_min > 0:
                # Set lower bound slightly below the min value
                lower_bound = max(0, y_min - (0.05 * (y_max - y_min)))  
                plt.ylim(lower_bound, y_max * 1.05)  # add a bit of headroom
            else:
                plt.ylim(y_min * 1.05, y_max * 1.05)

        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '\\' + save_path)

        if to_show:
            plt.show()


    def plot_overlaid_bar(
        self,
        x_axis: str,
        y_axis: str,
        fixed_value_tuple=None,
        ax=None,
        alpha=1.0,
        width=0.6,
        title="",
        y_label="",
        to_show=True,
        colors_for_value=(None, None)
    ):
        if ax is None:
            ax = plt.gca()

        (color_defining_column, color_dict) = colors_for_value

        if color_dict is not None:
            for value, color in color_dict.items():
                ax.bar(
                    0,
                    0,
                    color=color,
                    alpha=alpha,
                    label=str(value)
                )
        

        df = self.filter_dataframe(fixed_value_tuple)

        if df.empty:
            raise ValueError("No data to plot.")

        groups_by_x_axis = sorted(df[x_axis].unique())

        # Sort dataframe by y_axis values so bars go from lowest to highest
        df = df.sort_values(by=y_axis, ascending=False)

        for group_by_x_axis in groups_by_x_axis:

            sub_by_x_axis = df[df[x_axis] == group_by_x_axis]

            sub_by_x_axis = sub_by_x_axis.sort_values(
                by=y_axis,
                key=lambda col: col.abs(),
                ascending=False
            )

            for index, row in sub_by_x_axis.iterrows():

                color = None if color_defining_column is None else color_dict[row[color_defining_column]]

                ax.bar(
                    row[x_axis],
                    row[y_axis],
                    width=width,
                    alpha=alpha,
                    color=color
                )


        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_label or y_axis)
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

        if to_show:
            plt.show()

        return ax

    @requires_input_process
    def plot_graph(self, x_axis : str, y_axis : list, title : str = '', save_path: str = None, to_show=True, y_label=''):
        """
        Plots a graph using the dataframe stored in ResultLogger.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis.
        :param save_path: Optional path to save the plot as an image.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))


        if isinstance(y_axis, str):
            y_axis = [y_axis]
            
        for i in range(len(y_axis)):
            
            if isinstance(y_axis[i], str):
                y_axis[i] = (y_axis[i], y_axis[i])

            if y_axis[i][0] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i][0] }' not found in dataframe.")


        #plt.figure(figsize=(10, 6))
        for (column_name, name_to_plot) in y_axis:
            plt.plot(self.dataframe[x_axis], self.dataframe[column_name], marker='o', label=name_to_plot)

        plt.xlabel(x_axis)
        plt.ylabel(y_label)
        
        if title != '':
            plt.title(title)
                
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '\\' + save_path)

        if to_show:
            plt.show()


            
    @requires_input_process
    def plot_confidence_interval(self, x_axis : str, y_column : str, y_values_label : str = 'mean', show_std=True, alpha=0.3, 
                                 aggregate_number = None, title : str = '', save_path: str = None, to_show=True, 
                                 y_label='', x_slice_range=None, ax=None, color=None):

        if ax is None:
            ax = plt.gca()

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

        if y_column not in self.dataframe.columns:
            raise KeyError(f"Column '{y_column}' not found in dataframe. Available columns: " + str(self.dataframe.columns))

        x_values = self.dataframe[x_axis]
        values = self.dataframe[y_column]

        if x_slice_range is not None:
            x_values = x_values[x_slice_range[0]:max(x_slice_range[1], len(x_values))]
            values = values[x_slice_range[0]:max(x_slice_range[1], len(x_values))]

        if aggregate_number is not None:
            x_values, mean_values, std_values = aggregate_series(values, aggregate_number, show_std)

        else: # aggregate number is None
            mean_values = values
            std_values = None


        ax.plot(x_values, mean_values, label=y_values_label, color=color)

        if show_std and std_values is not None:
            ax.fill_between(x_values, mean_values - std_values, mean_values + std_values, alpha=alpha, color=color)


        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_label)

        if title != '':
            ax.set_title(title)

        ax.legend()
        ax.grid(True)

        if save_path:
            ax.figure.savefig(self.get_artifact_directory() + '\\' + save_path)

        if to_show:
            plt.show()

        return ax
    

    def convert_x_to_unique_int_numeric(self, x : list):

        if len(x) == 0:
            return []

        if isinstance(x[0], str):

            to_return = []

            already_seen_x_values = {}
            last_generated_value = -1

            for x_value in x:

                value = already_seen_x_values.get(x_value, None)

                if value is None:
                    last_generated_value += 1
                    already_seen_x_values[x_value] = last_generated_value
                    value = last_generated_value

                to_return.append(value)


            return to_return

        else:
            return x
    


    @requires_input_process
    def plot_linear_regression(self, x_axis: str, y_axis: str, title: str = '', save_path: str = None, to_show=True, y_label='', ax=None, fixed_value_tuple=None, color=None, group_by=None):

       if self.dataframe.empty:
           raise ValueError("Dataframe is empty. Log results before plotting.")

       if ax is None:
            ax = plt.gca()

       df = self.filter_dataframe(fixed_value_tuple)

       if group_by is not None:
            df = df.sort_values(by=group_by)
        
       x_plot, x_label_name, y_values, y_label_name = self.generate_values_from_dataframe(
           x_axis, y_axis, df
       )
   
       X_numeric = self.convert_x_to_unique_int_numeric(x_plot.values)

       print(f"X_numeric : {X_numeric}")


       if isinstance(X_numeric, list):
           X_numeric = np.array(X_numeric)

       if X_numeric.ndim == 1:
            X_numeric = X_numeric.reshape(-1, 1)
   
       Y_numeric = df[y_axis].values
   
       # ---- fit regression ----
       model = LinearRegression()
       model.fit(X_numeric, Y_numeric)
   
       Y_pred = model.predict(X_numeric)
   
       # ---- sort for clean line ----
       sort_idx = np.argsort(x_plot.astype(str))
       x_plot_sorted = np.array(x_plot)[sort_idx]
       y_pred_sorted = Y_pred[sort_idx]
   
       # ---- plot ----
       ax.plot(
           x_plot_sorted,
           y_pred_sorted,
           label=f"{y_axis} Regression",
           color=color
       )

       ax.set_xlabel(x_axis)
       ax.set_ylabel(y_label)

       if title != '':
            ax.set_title(title)

       ax.legend()
       ax.grid(True)

       if save_path:
           ax.figure.savefig(self.logDir + '\\' + save_path)

       if to_show:
           plt.show()


    def plot_polynomial_regression(
        self,
        x_axis: str,
        y_axis: list,
        degrees: int | list = 2,
        title: str = '',
        save_path: str = None,
        to_show=True,
        y_label='',
        color=None
    ):
        """
        Plots polynomial regression curves for the given columns in the dataframe.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis. Can be str or [(col_name, plot_label)].
        :param degrees: Polynomial degree or list of degrees (int or list[int]).
        :param title: Optional title for the plot.
        :param save_path: Optional path to save the plot as an image.
        :param to_show: Whether to display the plot.
        :param y_label: Label for the Y-axis.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: {list(self.dataframe.columns)}")

        if isinstance(y_axis, str):
            y_axis = [y_axis]

        for i in range(len(y_axis)):
            if isinstance(y_axis[i], str):
                y_axis[i] = (y_axis[i], y_axis[i])

            if y_axis[i][0] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i][0]}' not found in dataframe.")

        if isinstance(degrees, int):
            degrees = [degrees]  # normalize to list

        X = self.dataframe[[x_axis]].values
        X_sorted = np.sort(X, axis=0)  # ensures smooth curves

        for (column_name, name_to_plot) in y_axis:
            Y = self.dataframe[column_name].values

            for deg in degrees:
                # Create polynomial regression pipeline
                model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
                model.fit(X, Y)

                # Predict on sorted X for smooth curve
                Y_pred = model.predict(X_sorted)

                # Plot regression curve
                plt.plot(X_sorted, Y_pred, label=f'{name_to_plot} (degree {deg})', color=color)


        plt.xlabel(x_axis)
        plt.ylabel(y_label)

        if title:
            plt.title(title)

        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '\\' + save_path)

        if to_show:
            plt.show()

    @requires_input_process
    def plot_piecewise_linear_regression(self, x_axis: str, y_axis: list, n_segments: int,
                                     title: str = '', save_path: str = None, to_show=True, y_label='', plot_with_segment_names=False):
        """
        Plots a graph with piecewise linear regression lines for the given columns in the dataframe.

        :param x_axis: The column key for the X-axis.
        :param y_axis: A list of column keys for the Y-axis.
        :param n_segments: Number of segments to split the data into.
        :param title: Optional title for the plot.
        :param save_path: Optional path to save the plot as an image.
        :param to_show: Whether to display the plot.
        :param y_label: Label for the Y-axis.
        """
        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(f"Column '{x_axis}' not found in dataframe. Available columns: {self.dataframe.columns}")

        if isinstance(y_axis, str):
            y_axis = [y_axis]

        for i in range(len(y_axis)):
            if isinstance(y_axis[i], str):
                y_axis[i] = (y_axis[i], y_axis[i])

            if y_axis[i][0] not in self.dataframe.columns:
                raise KeyError(f"Column '{y_axis[i][0]}' not found in dataframe.")

        # Sort by x_axis to make clean splits
        df_sorted = self.dataframe.sort_values(by=x_axis).reset_index(drop=True)
        X_all = df_sorted[[x_axis]].values

        x_min, x_max = df_sorted[x_axis].min(), df_sorted[x_axis].max()
        segment_bounds = np.linspace(x_min, x_max, n_segments + 1)

        for (column_name, name_to_plot) in y_axis:
            Y_all = df_sorted[column_name].values

            ax = plt.gca()
            next_color = ax._get_lines.get_next_color()

            for i in range(n_segments):
                seg_mask = (df_sorted[x_axis] >= segment_bounds[i]) & (df_sorted[x_axis] < segment_bounds[i+1])
                X_seg = df_sorted.loc[seg_mask, [x_axis]].values
                Y_seg = df_sorted.loc[seg_mask, column_name].values

                if len(X_seg) < 2:
                    continue  # Skip segments without enough points

                # Fit linear regression for the segment
                model = LinearRegression()
                model.fit(X_seg, Y_seg)

                # Predict on the segment
                Y_pred = model.predict(X_seg)

                # Plot regression line
                if plot_with_segment_names:
                    plt.plot(X_seg, Y_pred, color=next_color, label=f'{name_to_plot} Seg {i+1}')
                else:   
                    plt.plot(X_seg, Y_pred, color=next_color)

        plt.xlabel(x_axis)
        plt.ylabel(y_label)

        if title:
            plt.title(title)

        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(self.logDir + '/' + save_path)

        if to_show:
            plt.show()


    def plot_current_graph(self, title: str = None, y_label : str = None, y_min =None):
        if title is not None:
            plt.title(title)

        if y_label != None:
            plt.ylabel(ylabel=y_label)

        # Apply minimum y-axis limit if provided
        if y_min is not None:
            current_ylim = plt.ylim()          # (current_min, current_max)
            plt.ylim(bottom=y_min, top=current_ylim[1])

        plt.show()

    @requires_input_process
    def plot_covariance_and_correlation(
        self,
        x_axis: str,
        y_axis: str,
        title: str = '',
        save_path: str = None,
        to_show: bool = True,
        fixed_value_tuple=None,
        ax=None,
        color=None,
        show_values_in_title: bool = True
    ):
        """
        Plots a scatter plot between two dataframe columns and computes both covariance and correlation.

        :param x_axis: Column name for the X-axis.
        :param y_axis: Column name for the Y-axis.
        :param title: Optional plot title.
        :param save_path: Optional path to save the plot as an image.
        :param to_show: Whether to display the plot.
        :param fixed_value_tuple: Optional filters like [("column_name", value)].
        :param ax: Optional matplotlib axis.
        :param color: Optional scatter color.
        :param show_values_in_title: Whether to append covariance and correlation to the title.
        :return: (covariance, correlation)
        """

        if self.dataframe.empty:
            raise ValueError("Dataframe is empty. Log results before plotting.")

        if x_axis not in self.dataframe.columns:
            raise KeyError(
                f"Column '{x_axis}' not found in dataframe. "
                f"Available columns: {self.dataframe.columns}"
            )

        if y_axis not in self.dataframe.columns:
            raise KeyError(
                f"Column '{y_axis}' not found in dataframe. "
                f"Available columns: {self.dataframe.columns}"
            )

        if ax is None:
            ax = plt.gca()

        df = self.filter_dataframe(fixed_value_tuple)

        if df.empty:
            raise ValueError("No data available after applying filters.")

        # Ensure numeric values
        x_values = pd.to_numeric(df[x_axis], errors="coerce")
        y_values = pd.to_numeric(df[y_axis], errors="coerce")

        valid_mask = x_values.notna() & y_values.notna()
        x_values = x_values[valid_mask]
        y_values = y_values[valid_mask]

        if len(x_values) < 2:
            raise ValueError("At least two valid numeric rows are required.")

        # ---- statistics ----
        covariance = x_values.cov(y_values)
        correlation = x_values.corr(y_values)

        # ---- plot ----
        ax.scatter(
            x_values,
            y_values,
            color=color,
            label=f"Cov = {covariance:.4f}, Corr = {correlation:.4f}"
        )

        # ---- title ----
        final_title = title
        if show_values_in_title:
            stats_text = f"Cov = {covariance:.4f}, Corr = {correlation:.4f}"
            final_title = f"{title} - {stats_text}" if title else stats_text

        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)

        if final_title:
            ax.set_title(final_title)

        ax.legend()
        ax.grid(True)

        if save_path:
            ax.figure.savefig(self.logDir + '\\' + save_path)

        if to_show:
            plt.show()

        return covariance, correlation


    @requires_input_process
    def list_of_unique_values(self, column):

        return self.dataframe[column].unique().tolist()


 
def get_results_logger_from_file(folder_path, results_filename=RESULTS_FILENAME) -> ResultLogger:
        
    artifact_base_directory = folder_path
    artifact_relative_directory = ''
    
    datafrane = pd.read_csv(os.path.join(artifact_base_directory, artifact_relative_directory, results_filename))
    
    results_columns = datafrane.columns.tolist()    
    
    resuls_logger = ResultLogger(
        {
            "artifact_relative_directory": artifact_relative_directory,
            "base_directory": artifact_base_directory,
            "create_new_directory": False,
            "results_filename": results_filename,
            "results_columns": results_columns
        }
    )
    
    return resuls_logger
        
            
def aggregate_results_logger_from_paths(paths, new_directory, new_column=None, results_filename=RESULTS_FILENAME, new_results_filename=None, parent_component : Component =None) -> ResultLogger:
    
    '''
    new_column is a tuple (column_name, [values for each dataframe loaded])
    '''
    if new_results_filename == None:
        new_results_filename = results_filename


    datafrane = None
    
    for folder_path_index in range(len(paths)):
        
        folder_path = paths[folder_path_index]
        
        artifact_base_directory = folder_path
        artifact_relative_directory = ''

        if datafrane is None:
            datafrane = pd.read_csv(os.path.join(artifact_base_directory, artifact_relative_directory, results_filename))
            
            if new_column is not None:
                datafrane[new_column[0]] = new_column[1][folder_path_index]
                            
        else:
            
            loaded_dataframe = pd.read_csv(os.path.join(artifact_base_directory, artifact_relative_directory, results_filename))

            if new_column is not None:
                loaded_dataframe[new_column[0]] = new_column[1][folder_path_index]
                            
            datafrane = pd.concat([datafrane, loaded_dataframe])

    results_columns = datafrane.columns.tolist() 
    
    resuls_logger = ResultLogger(
        {
            "artifact_relative_directory": '',
            "base_directory": new_directory,
            "create_new_directory": False,
            "results_filename": new_results_filename,
            "results_columns": results_columns
        }
    )
    
    resuls_logger.saveDataframe(datafrane, filename=new_results_filename)
    
    return resuls_logger
    

def aggregate_results_logger(results_logger_objects : list[ResultLogger], new_directory, new_column=None, new_results_filename=RESULTS_FILENAME, parent_component : Component =None) -> ResultLogger:
    
    '''
    new_column is a tuple (column_name, [values for each dataframe loaded])
    '''

    datafrane = None
    
    for results_logger_index in range(len(results_logger_objects)):
        
        results_logger : ResultLogger = results_logger_objects[results_logger_index]

        if datafrane is None:
            datafrane = results_logger.get_dataframe()
            
            if new_column is not None: # if we're going to add a new column

                datafrane[new_column[0]] = new_column[1][results_logger_index]
                            
        else:
            
            loaded_dataframe = results_logger.get_dataframe()

            if new_column is not None:
                loaded_dataframe[new_column[0]] = new_column[1][results_logger_index]
                            
            datafrane = pd.concat([datafrane, loaded_dataframe])

    results_columns = datafrane.columns.tolist() 
    
    resuls_logger = ResultLogger(
        {
            "artifact_relative_directory": '',
            "base_directory": new_directory,
            "create_new_directory": False,
            "results_filename": new_results_filename,
            "results_columns": results_columns
        }
    )
    
    resuls_logger.saveDataframe(datafrane, filename=new_results_filename)
    
    return resuls_logger
    
    

