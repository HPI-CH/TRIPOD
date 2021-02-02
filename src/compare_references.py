"""
This script compares different reference systems against each other.
"""

from pipeline.reference_loader import ZebrisReferenceLoader, OptogaitReferenceLoader
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from matplotlib.lines import Line2D


def reg_line(x, y):
    """
    Calculate regression line for data.

    Args:
        x (list[float]): data x values
        y (list[float]): data y values

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
    # print('p values:')
    # print(results.pvalues)
    # print('confidence intervals:')
    # print(conf_interval)

    # calculate RMSE (root mean squared error)
    y_pred = gradient * x + intercept
    rmse = np.sqrt(np.mean((y_pred - y) ** 2))

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
    ]

    return x1, y1, line_info, pvalues, conf_interval


def detect_outlier(
    data, column, reference_column, z_threshold=3, maximum_deviation=0.25
):
    """
    Detect outliers based on z-score and maximum deviation.

    Args:
        data (DataFrame): DataFrame with the data
        column (str): name of the column under consideration
        reference_column (str): name of the reference column matching the data column
        z_threshold (float): cutoff value for z-score
        maximum_deviation (float): cutoff value for deviation between systems

    Returns:
        np.array: boolean vector indicating outliers
    """
    column_diff = np.abs(data[column] - data[reference_column])
    z_score_diff = np.abs(stats.zscore(column_diff))
    # outlier = np.logical_or(z_score_diff > z_threshold, column_diff > maximum_deviation)
    outlier = z_score_diff > z_threshold
    return outlier


def draw_reg_line_and_info(data, outlier, column, reference_column, axis):
    """
    Plot regression line and parameters.

    Args:
        data (DataFrame): DataFrame with actual data
        outliers (np.array): Boolean array, indicating if the row is an outlier
        column (str): name of the column under consideration
        reference_column (str): name of the reference column matching the data column
        axis (matplotlib.axis): axis to plot on

    Returns:
        None
    """

    x1, y1, info, pvalues, conf_interval = reg_line(
        data[np.logical_not(outlier)][column],
        data[np.logical_not(outlier)][reference_column],
    )

    textstr = "\n".join(
        (
            r"$n=%i$" % (len(data[np.logical_not(outlier)]),),
            r"$r=%.2f$" % (info[2],),
            r"$RMSE=%.2f$" % (info[5],),
            r"$y=%.2fx %+.2f$" % (info[0], info[1]),
        )
    )

    props = dict(boxstyle="square", facecolor="white", edgecolor="white", alpha=0)

    axis.text(
        0.97,
        0.03,
        textstr,
        fontsize=11,
        transform=axis.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    axis.plot(x1, y1)
    axis.plot(x1, x1, color="0.75")


if __name__ == "__main__":
    config = {
        "name": "manual_threshold",
        # "raw_base_path": "./data/raw",
        "raw_base_path": "./example_data_/raw",
        # "interim_base_path": "./data/interim",
        "interim_base_path": "./example_data/interim",
        "overwrite": False,
        # "dataset": "TRIPOD",
        "dataset": "TRIPOD_excerpt",
        "subjects": [
            # "Sub_DF",
            # "Sub_HA",
            # "Sub_PB",
            # "Sub_AL",
            # "Sub_EN",
            # "Sub_KP",
            # "Sub_RW",
            # "Sub_BK",
            "Sub_FZ",
            # "Sub_LU",
            # "Sub_SN",
            # "Sub_CP",
            # "Sub_GK",
            # "Sub_OD",
            # "Sub_YU",
        ],
        "runs": ["PWS", "PWS+20", "PWS-20"],
        "experiment_duration": 120,
        "plot_outlier": False,
    }

    plt.rcParams.update({"font.size": 12})

    cmap = matplotlib.cm.get_cmap("jet")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=len(config["subjects"]) - 0.99)

    all_data = pd.DataFrame()

    fig_sl, ax_sl = plt.subplots()  # stride length
    fig_st, ax_st = plt.subplots()  # stride time
    fig_swt, ax_swt = plt.subplots()  # swing time
    fig_stt, ax_stt = plt.subplots()  # stance time
    fig_td, ax_td = plt.subplots()  # timestamp difference

    for subject_id, subject in enumerate(config["subjects"]):
        for run_id, run in enumerate(config["runs"]):

            zebris_data = ZebrisReferenceLoader(
                config["name"],
                config["raw_base_path"],
                config["interim_base_path"],
                config["dataset"],
                subject,
                run,
                config["overwrite"],
            ).get_data()

            optogait_data = OptogaitReferenceLoader(
                config["name"],
                config["raw_base_path"],
                config["interim_base_path"],
                config["dataset"],
                subject,
                run,
                config["overwrite"],
            ).get_data()

            merged = {"left": None, "right": None}
            for side in ["left", "right"]:
                zebris_data[side]["timestamp_zebris"] = zebris_data[side]["timestamp"]
                merged[side] = pd.merge_asof(
                    left=optogait_data[side],
                    right=zebris_data[side],
                    on="timestamp",
                    direction="nearest",
                    tolerance=0.1,
                    allow_exact_matches=True,
                ).dropna()
                merged[side] = merged[side][
                    merged[side]["timestamp"] < config["experiment_duration"]
                ]
                merged[side]["subject"] = subject_id
                merged[side]["run"] = run_id

                all_data = pd.concat([all_data, merged[side]])

    # Compute z scores for outlier
    stride_length_outlier = detect_outlier(
        all_data, "stride_length_ref_y", "stride_length_ref_x"
    )
    stride_time_outlier = detect_outlier(
        all_data, "stride_time_ref_y", "stride_time_ref_x"
    )
    swing_time_outlier = detect_outlier(
        all_data, "swing_time_ref_y", "swing_time_ref_x"
    )
    stance_time_outlier = detect_outlier(
        all_data, "stance_time_ref_y", "stance_time_ref_x"
    )

    # outlier = np.logical_or(stance_time_outlier, np.logical_or(swing_time_outlier, np.logical_or(stride_length_outlier, stride_time_outlier)))
    # stride_length_outlier = outlier
    # stride_time_outlier = outlier
    # swing_time_outlier = outlier
    # stance_time_outlier = outlier

    for subject_id, subject in enumerate(config["subjects"]):
        for run_id, run in enumerate(config["runs"]):

            rgba = cmap(norm(subject_id))
            # if run_id == 0:
            #     marker = "o"
            # elif run_id == 1:
            #     marker = "<"
            # elif run_id == 2:
            #     marker = ">"
            marker = "."

            mask = np.logical_and(
                all_data["subject"] == subject_id, all_data["run"] == run_id
            )
            plot_data = all_data[mask]

            outlier = (
                np.zeros_like(stride_length_outlier[mask])
                if config["plot_outlier"]
                else stride_length_outlier[mask]
            )
            ax_sl.scatter(
                plot_data["stride_length_ref_y"][np.logical_not(outlier)],
                plot_data["stride_length_ref_x"][np.logical_not(outlier)],
                color=rgba,
                marker=marker,
                s=4 ** 2,
                alpha=0.2,
            )
            outlier = (
                np.zeros_like(stride_time_outlier[mask])
                if config["plot_outlier"]
                else stride_time_outlier[mask]
            )
            ax_st.scatter(
                plot_data["stride_time_ref_y"][np.logical_not(outlier)],
                plot_data["stride_time_ref_x"][np.logical_not(outlier)],
                color=rgba,
                marker=marker,
                s=4 ** 2,
                alpha=0.2,
            )
            outlier = (
                np.zeros_like(swing_time_outlier[mask])
                if config["plot_outlier"]
                else swing_time_outlier[mask]
            )
            ax_swt.scatter(
                plot_data["swing_time_ref_y"][np.logical_not(outlier)],
                plot_data["swing_time_ref_x"][np.logical_not(outlier)],
                color=rgba,
                marker=marker,
                s=4 ** 2,
                alpha=0.2,
            )
            outlier = (
                np.zeros_like(stance_time_outlier[mask])
                if config["plot_outlier"]
                else stance_time_outlier[mask]
            )
            ax_stt.scatter(
                plot_data["stance_time_ref_y"][np.logical_not(outlier)],
                plot_data["stance_time_ref_x"][np.logical_not(outlier)],
                color=rgba,
                marker=marker,
                s=4 ** 2,
                alpha=0.2,
            )
            ax_td.scatter(
                plot_data["timestamp"],
                plot_data["timestamp"] - plot_data["timestamp_zebris"],
                color=rgba,
                marker=marker,
                s=4 ** 2,
                alpha=0.2,
            )

    if config["plot_outlier"]:
        # plot outliers
        ax_sl.scatter(
            all_data[stride_length_outlier]["stride_length_ref_y"],
            all_data[stride_length_outlier]["stride_length_ref_x"],
            marker="o",
            s=7 ** 2,
            facecolors="None",
            edgecolors="r",
        )
        ax_st.scatter(
            all_data[stride_time_outlier]["stride_time_ref_y"],
            all_data[stride_time_outlier]["stride_time_ref_x"],
            marker="o",
            s=7 ** 2,
            facecolors="None",
            edgecolors="r",
        )
        ax_swt.scatter(
            all_data[swing_time_outlier]["swing_time_ref_y"],
            all_data[swing_time_outlier]["swing_time_ref_x"],
            marker="o",
            s=7 ** 2,
            facecolors="None",
            edgecolors="r",
        )
        ax_stt.scatter(
            all_data[stance_time_outlier]["stance_time_ref_y"],
            all_data[stance_time_outlier]["stance_time_ref_x"],
            marker="o",
            s=7 ** 2,
            facecolors="None",
            edgecolors="r",
        )

    # calculate stride lenght slope
    draw_reg_line_and_info(
        all_data,
        stride_length_outlier,
        "stride_length_ref_y",
        "stride_length_ref_x",
        ax_sl,
    )
    draw_reg_line_and_info(
        all_data, stride_time_outlier, "stride_time_ref_y", "stride_time_ref_x", ax_st
    )
    draw_reg_line_and_info(
        all_data, swing_time_outlier, "swing_time_ref_y", "swing_time_ref_x", ax_swt
    )
    draw_reg_line_and_info(
        all_data, stance_time_outlier, "stance_time_ref_y", "stance_time_ref_x", ax_stt
    )

    legend_elements = [
        Line2D([0], [0], color="0.75", label="unity slope"),
        Line2D([0], [0], label="regression line"),
    ]
    if config["plot_outlier"]:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker="o",
                markerfacecolor="None",
                markeredgecolor="r",
                label="outlier",
            )
        )

    # ax_sl.set_title("Reference comparison - stride length")
    ax_sl.set_xlabel("Zebris stride length [m]")
    ax_sl.set_ylabel("Optogait stride length [m]")
    ax_sl.set_aspect("equal")
    ax_sl.legend(handles=legend_elements, loc="upper left")

    # ax_st.set_title("Reference comparison - stride time")
    ax_st.set_xlabel("Zebris stride time [s]")
    ax_st.set_ylabel("Optogait stride time [s]")
    ax_st.set_aspect("equal")
    ax_st.legend(handles=legend_elements, loc="upper left")

    # ax_td.set_title("Reference comparison - timestamp missmatch")
    ax_td.set_xlabel("Zebris timestamp [s]")
    ax_td.set_ylabel("Difference OptoGait - Zebris timestamp [s]")

    # ax_swt.set_title("Reference comparison - swing time")
    ax_swt.set_xlabel("Zebris swing time [s]")
    ax_swt.set_ylabel("Optogait swing time [s]")
    ax_swt.set_aspect("equal")
    ax_swt.legend(handles=legend_elements, loc="upper left")

    # ax_stt.set_title("Reference comparison - stance time")
    ax_stt.set_xlabel("Zebris stance time [s]")
    ax_stt.set_ylabel("Optogait stance time [s]")
    ax_stt.set_aspect("equal")
    ax_stt.legend(handles=legend_elements, loc="upper left")

    plt.show()
