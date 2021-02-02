"""This module contains the implementation of the Evaluator."""

import pandas as pd
import numpy as np
import json
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import mpld3
from matplotlib.lines import Line2D


class Evaluator:
    """
    The Evaluator merges temporal and spatial gait parameters
    from the trajectory estimation algorithm, the gait event
    detection algorithm and the reference system and creates
    plots and metrics in order to assess the quality of the
    used algorithms and data.

    The pipeline is meant to be executed for each trial individually.
    The results of the pipeline are then added to the Evaluator using
    add_data().
    """

    def __init__(self):
        self.data = {}
        self.unit_for_column = {
            "stride_length": "m",
            "stride_time": "s",
            "swing_time": "s",
            "stance_time": "s",
        }

    def add_data(self, subject_num, run_num, data, reference_data):
        """
        Adds a new pair of IMU and reference data for one trial.

        Args:
            subject_num (int): Subject index
            run_num (int): Run index
            data (dict[str, DataFrame]): Gait parameters produced by the pipeline for each foot
            reference_data (dict[str, DataFrame]): Reference gait parameters for each foot

        Returns:
            None
        """
        self.data[(subject_num, run_num)] = {
            "data": data,
            "reference_data": reference_data,
            "merged": {},
        }

    def save(self, filename):
        """
        Save the evaluation results to a json file.
        This can be handy if different plots should be produced
        without rerunning the whole pipeline.

        Args:
            filename (str): filename for the saved data

        Returns:
            None
        """
        serializable_data = []
        for key, value in self.data.items():
            serializable_data.append(
                [
                    key[0],
                    key[1],
                    value["merged"]["left"].to_json(),
                    value["merged"]["right"].to_json(),
                ]
            )

        json.dump(serializable_data, open(filename, "w"))

    def load(self, filename):
        """
        Load the evaluation results from a json file.

        Args:
            filename (str): filename of the saved data

        Returns:
            None
        """
        serialized_data = json.load(open(filename, "r"))
        for entry in serialized_data:
            self.data[(entry[0], entry[1])] = {
                "merged": {
                    "left": pd.read_json(entry[2]),
                    "right": pd.read_json(entry[3]),
                }
            }

    def match_timestamps(self):
        """
        Match timestamps from reference system and IMUs.
        All timestamps are zero based to the timestamp of initial contact.
        In order to match gait parameters form the reference system and IMUs
        stride by stride, they are merged based on minimal time difference
        of the timestamps within a certain tolerance.
        Strides that cannot be matched are dropped.

        Returns:
            None
        """
        for data in self.data.values():
            for side in ["left", "right"]:
                data["reference_data"][side]["timestamp_ref"] = data["reference_data"][
                    side
                ]["timestamp"]
                data["merged"][side] = pd.merge_asof(
                    left=data["data"][side],
                    right=data["reference_data"][side],
                    on="timestamp",
                    direction="nearest",
                    tolerance=0.3,
                    allow_exact_matches=True,
                ).dropna()

    def detect_outliers(self, column):
        """
        Detect outliers via z-score and maximum deviation for one column.
        Marks the outliers in a special boolean column.

        Args:
            column (str): Name of the column under consideration (stride_length, stride_time, swing_time, stance_time).

        Returns:
            None
        """
        z_threshold = 3
        maximum_deviation = {
            "stride_length": 0.25,
            "stride_time": 0.3,
            "swing_time": 0.3,
            "stance_time": 0.3,
        }
        # Compute z scores
        reference_column = column + "_ref"
        for data in self.data.values():
            for side in ["left", "right"]:
                data["merged"][side]["column_diff"] = np.abs(
                    data["merged"][side][column]
                    - data["merged"][side][reference_column]
                )
                data["merged"][side]["z_score_diff"] = np.abs(
                    stats.zscore(data["merged"][side]["column_diff"])
                )
                data["merged"][side]["outlier"] = np.logical_or(
                    data["merged"][side]["z_score_diff"] > z_threshold,
                    data["merged"][side]["column_diff"] > maximum_deviation[column],
                )

    def merge_subject_runs(self, subject_run_nums):
        """
        Merge data from all subjects and runs into one DataFrame and
        add special columns for subject and run.

        Args:
            subject_run_num (list[tuple[int, int]]): list of subject and run numbers that should be merged

        Returns:
            dict[str, DataFrame]: merged DataFrame for each foot
        """
        merged = {}
        for side in ["left", "right"]:
            data_with_sub_run_num = []
            for subject_run_num in subject_run_nums:
                self.data[subject_run_num]["merged"][side]["subject"] = subject_run_num[
                    0
                ]
                self.data[subject_run_num]["merged"][side]["run"] = subject_run_num[1]
                data_with_sub_run_num.append(self.data[subject_run_num]["merged"][side])

            merged[side] = pd.concat(data_with_sub_run_num)

        return merged

    def merge_sides(self, merged_subject_runs):
        """
        Merge data from left and right foot.

        Args:
            merged_subject_runs (dict[str, DataFrame]): DataFrames for left and right foot

        Returns:
            DataFrame: Merged DataFrame for both, left and right foot
        """
        return pd.concat((merged_subject_runs["left"], merged_subject_runs["right"]))

    def filter_outliers(self, data):
        """
        Select all datapoints that have not been labeled previously as outliers by detect_outliers().

        Args:
            data (DataFrame): DataFrame with outliers marked in column "outlier"

        Returns:
            DataFrame: Filtered DataFrame without outliers
        """

        return data[data["outlier"] == False]

    def reg_line(self, x, y):
        """
        Calculate linear regression on two dimensional data.

        Args:
            x (list[float]): x values of the data
            y (list[float]): y values of the data

        Returns:
            tuple(list[float], ...): x values of the regression line, y values of the regression line, calculated regression line parameters (gradient, intercept, r_value, p_value, std_err, rmse, mae), p-values of the statistical model, confidence interval
        """
        gradient, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # get p values and CI for the gradient and intercept
        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        results = model.fit()
        pvalues = results.pvalues
        conf_interval = results.conf_int(alpha=0.05, cols=None)

        # calculate RMSE (root mean squared error)
        y_pred = gradient * x + intercept
        rmse = np.sqrt(np.mean((y_pred - y) ** 2))
        mae = np.mean(np.abs(y_pred - y))

        # make a regression line
        mn = np.min(x)
        mx = np.max(x + 0.5)
        mn = 0
        x1 = np.linspace(mn, mx, 500)
        y1 = gradient * x1 + intercept

        # summary line info
        line_info = [
            round(gradient, 4),
            round(intercept, 4),
            round(r_value, 4),
            round(p_value, 4),
            round(std_err, 4),
            round(rmse, 4),
            round(mae, 4),
        ]

        return x1, y1, line_info, pvalues, conf_interval

    def plot_correlation(self, title, column, subject_run_nums, reference_name):
        """
        Create correlation plot of IMU data and reference data for the selected subjects and runs.

        Args:
            title (str): Title of the plot
            column (str): Name of the data column that should be plotted
            subject_run_nums (list[tuple[int, int]]): List of subject and run indices
            reference_name (str): Name of the reference system (used as axis label)

        Returns:
            None
        """
        self.detect_outliers(column)

        merged = self.merge_sides(self.merge_subject_runs(subject_run_nums))

        merged = self.filter_outliers(merged)

        reference_column = column + "_ref"

        x = merged[reference_column]
        y = merged[column]

        axes_min = np.minimum(np.min(x), np.min(y))
        axes_max = np.maximum(np.max(x), np.max(y))
        x_line, y_line, info, pvalues, conf_interval = self.reg_line(x, y)

        fig, ax = plt.subplots(figsize=(5, 5))
        plt.rcParams.update({"font.size": 12})

        unit = self.unit_for_column[column]
        plt.xlabel(reference_name + " " + column.replace("_", " ") + " [" + unit + "]")
        plt.ylabel("IMU " + column.replace("_", " ") + " [" + unit + "]")

        legend_elements = [
            Line2D([0], [0], color="0.75", label="unity slope"),
            Line2D([0], [0], label="regression line"),
        ]
        plt.legend(handles=legend_elements, loc=2)

        print("r=", info[2])
        print(f"y={info[0]}x + {info[1]}")

        # plt.title(title + " " + column.replace("_", " "))

        textstr = "\n".join(
            (
                r"$n=%i$" % (len(x),),
                r"$r=%.2f$" % (info[2],),
                r"$RMSE=%.2f$" % (info[5],),
                # r'$MAE=%.2f$' % (info[6], ),
                r"$y=%.2fx %+.2f$" % (info[0], info[1]),
            )
        )

        props = dict(boxstyle="square", facecolor="white", edgecolor="white", alpha=0)

        # place a text box in bottom right in axes coords
        ax.text(
            0.97,
            0.03,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        ax.plot(x_line, y_line)
        ax.plot(x_line, x_line, color="0.75")
        scatters = []
        labels = []
        for (subject, run) in subject_run_nums:
            scatters.append(
                ax.scatter(
                    x[
                        np.logical_and(
                            merged["subject"] == subject, merged["run"] == run
                        )
                    ],
                    y[
                        np.logical_and(
                            merged["subject"] == subject, merged["run"] == run
                        )
                    ],
                    marker="o",
                    s=3 ** 2,
                    alpha=0.2,
                )
            )
            labels.append(
                [
                    "t={:.2f}".format(t)
                    + "\nSubject "
                    + str(subject)
                    + " Run "
                    + str(run)
                    for t in merged[
                        np.logical_and(
                            merged["subject"] == subject, merged["run"] == run
                        )
                    ]["timestamp"]
                ]
            )

        tooltips = [
            mpld3.plugins.PointLabelTooltip(scatter, labels=label)
            for scatter, label in zip(scatters, labels)
        ]

        for tooltip in tooltips:
            mpld3.plugins.connect(fig, tooltip)

        plt.xlim([axes_min - 0.05 * (axes_max - axes_min), axes_max + 0.05 * (axes_max - axes_min)])
        plt.ylim([axes_min - 0.05 * (axes_max - axes_min), axes_max + 0.05 * (axes_max - axes_min)])

        # mpld3.show()
        #plt.axis("square")
        plt.show()
        # plt.savefig(title)
        # plt.close()

    def plot_bland_altmann(self, column, subject_run_nums, reference_name):
        """
        Create Bland-Altman plots of IMU data and reference data for the selected subjects and runs.

        Args:
            column (str): Name of the data column that should be plotted
            subject_run_nums (list[tuple[int, int]]): List of subject and run indices
            reference_name (str): Name of the reference system (used as axis label)

        Returns:
            None
        """
        sd_limit = 1.96
        confidence = 0.95

        unit = self.unit_for_column[column]

        self.detect_outliers(column)

        merged = self.merge_sides(self.merge_subject_runs(subject_run_nums))

        merged = self.filter_outliers(merged)

        reference_column = column + "_ref"

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.rcParams.update({"font.size": 12})

        n = merged[column].size
        diff = merged[column] - merged[reference_column]
        mean_between_methods = (merged[column] + merged[reference_column]) / 2
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, axis=0)

        for (subject, run) in subject_run_nums:
            mask = np.logical_and(merged["subject"] == subject, merged["run"] == run)
            ax.scatter(
                merged[reference_column][mask],
                diff[mask],
                marker="o",
                s=3 ** 2,
                alpha=0.2,
            )

        right_xlim = np.amax(merged[reference_column]) + 0.5 * (
            np.amax(merged[reference_column]) - np.amin(merged[reference_column])
        )

        if confidence is not None:
            assert 0 < confidence < 1
            ci = dict()
            ci["mean"] = stats.norm.interval(
                confidence, loc=mean_diff, scale=std_diff / np.sqrt(n)
            )
            seLoA = ((1 / n) + (sd_limit ** 2 / (2 * (n - 1)))) * (std_diff ** 2)
            loARange = np.sqrt(seLoA) * stats.t.ppf((1 - confidence) / 2, n - 1)
            ci["upperLoA"] = (
                (mean_diff + sd_limit * std_diff) + loARange,
                (mean_diff + sd_limit * std_diff) - loARange,
            )
            ci["lowerLoA"] = (
                (mean_diff - sd_limit * std_diff) + loARange,
                (mean_diff - sd_limit * std_diff) - loARange,
            )
            # print(ci["upperLoA"])
            # print(ci["lowerLoA"])

        # Plot the SD intervals as horizontal lines
        if sd_limit > 0:
            # half_ylim = (1.5 * sd_limit) * std_diff
            # ax.set_ylim(mean_diff - half_ylim,
            # mean_diff + half_ylim)
            limit_of_agreement = sd_limit * std_diff
            lower = mean_diff - limit_of_agreement
            upper = mean_diff + limit_of_agreement
            for j, lim in enumerate([lower, upper]):
                ax.axhline(lim, color="0.75")  # ,linestyle=':')

            ax.annotate(
                "-{}SD: {:.2f}".format(sd_limit, np.round(lower, 2)),
                xy=(right_xlim - 0.01, lower - 0.01),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=10,
                xycoords="data",
            )
            ax.annotate(
                "+{}SD: {:.2f}".format(sd_limit, np.round(upper, 2)),
                xy=(right_xlim - 0.01, upper - 0.01),
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=10,
                xycoords="data",
            )

        # Plot a horizontal line at mean difference
        ax.axhline(mean_diff, ls="-", c="0.75")

        ax.annotate(
            "Mean: {:.2f}".format(np.round(mean_diff, 2)),
            xy=(right_xlim - 0.01, mean_diff - 0.01),
            horizontalalignment="right",
            verticalalignment="top",
            fontsize=10,
            xycoords="data",
        )

        ax.set_xlim(right=right_xlim)
        ax.set_xlabel(
            "Average of the two methods [" + unit + "]"
        )
        ax.set_ylabel(
            "IMU - "
            + reference_name
            + " "
            + column.replace("_", " ")
            + " ["
            + unit
            + "]"
        )

        textstr = "\n".join(
            (
                r"$n=%i$" % (n,),
                r"$LoA=%.2f %s$" % (limit_of_agreement, unit),
            )
        )

        props = dict(boxstyle="square", facecolor="white", edgecolor="white", alpha=0)

        # place a text box in bottom right in axes coords
        ax.text(
            0.97,
            0.03,
            textstr,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        plt.show()
