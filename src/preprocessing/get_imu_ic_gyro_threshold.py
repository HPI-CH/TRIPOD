"""
Two different kinds of information need to get extracted from the IMU recordings only
once and are therefore outsourced to this preprocessing script:
The timestamp of initial contact and
the gyroscope magnitude threshold that identifies stance phases best.

In order to synchronize Optogait and/or Zebris recordings with the IMU recordings,
the initial contact with the treadmill belt has to be found in the IMU data.
Since the participants were asked to step on the running treadmill belt with their
right foot first, the first peak in the acceleration data of the right foot corresponds
to the initial contact.
Therefore this script:
Opens all right foot IMU CSV file in the base path,
Finds first prominent peak (first peak with prominence > 4, this threshold is chosen to
fit the recorded data and might not be the best choice for other data recordings),
Plot data to let user confirm/correct the peak by manually placing the moment of
initial contact (e.g. by experience or with the help of the video recordings), and
Exports a CSV file with the obtained timesteps

Low gyroscope energy has turned out to be a good indicator for stance phases. However,
the threshold to identify stance phases is different for each subject and trial.
Therefore this script:
Calculates stance phases based on a default threshold,
Displays them, and let the user confirm/correct the threshold, and
Exports a CSV file with the thresholds.
"""

import os
import sys

from data_reader.imu import IMU
from event_detection.imu_event_detection import gyro_threshold_stance
from visualization.plot import plot_gyro_magnitude, show
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal
import pandas as pd

import warnings


def ic_onclick(event):
    """ Event handler for click event """
    global select_vline
    global selection_x_coordinate
    if select_vline is not None:
        select_vline.remove()
    select_vline = plt.axvline(event.xdata, color="b")
    selection_x_coordinate = event.xdata
    plt.draw()


def stance_onclick(event):
    """ Event handler for click event """
    global stance_magnitude_threshold
    global stance_count_threshold
    global imu
    global fig

    stance_magnitude_threshold = event.ydata
    stance = gyro_threshold_stance(
        imu,
        stance_magnitude_threshold=stance_magnitude_threshold,
        stance_count_threshold=stance_count_threshold,
    )
    plt.close(fig=fig)
    fig = plot_gyro_magnitude(imu, stance=stance, threshold=stance_magnitude_threshold)

    fig.canvas.mpl_connect("key_press_event", stance_onclick)
    show()


if __name__ == "__main__":
    sys.path.append("./src/")
    warnings.simplefilter("error", RuntimeWarning)

    select_vline = None
    selection_x_coordinate = None

    base_path = "./data/raw/TRIPOD"

    # set flags according on what acction should be performed
    check_timestamps = True
    check_stance_phase_manually = False

    ic_timestamps = []
    gyro_thresholds = []

    subjects = [
        x
        for x in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, x)) and x.startswith("Sub")
    ]

    for subject_id, subject in enumerate(subjects):
        subject_directory = os.path.join(base_path, subject)
        runs = [
            x
            for x in os.listdir(subject_directory)
            if os.path.isdir(os.path.join(subject_directory, x))
        ]
        for run_id, run in enumerate(runs):
            print(
                "subject",
                subject_id + 1,
                "/",
                len(subjects),
                "run",
                run_id + 1,
                "/",
                len(runs),
            )
            run_directory = os.path.join(subject_directory, run)

            if check_timestamps:
                right_foot_path = os.path.join(run_directory, "IMU", "RF.csv")

                imu = IMU(right_foot_path)

                imu.check_sampling()
                exit()

                time = imu.time()
                accel = np.transpose(imu.accel())
                accel_norm = np.linalg.norm(accel, axis=0)

                prominence_threshold = 3

                peaks, _ = sp.signal.find_peaks(accel_norm)
                prominences = sp.signal.peak_prominences(accel_norm, peaks)[0]

                first_prominent_peak = time[
                    peaks[prominences > prominence_threshold][0]
                ]
                selection_x_coordinate = first_prominent_peak

                fig = plt.figure()

                fig.canvas.mpl_connect("key_press_event", ic_onclick)

                plt.plot(time, accel[0], label="X")
                plt.plot(time, accel[1], label="Y")
                plt.plot(time, accel[2], label="Z")
                plt.axvline(first_prominent_peak, color="r")
                plt.xlabel("Time [s]")
                plt.ylabel("IMU Acceleration [g]")
                plt.legend()
                plt.show()

                selection_x_coordinate = time[
                    np.argmin(np.abs(time - selection_x_coordinate))
                ]
                print(selection_x_coordinate)

                ic_timestamps.append([subject, run, selection_x_coordinate])

            if check_stance_phase_manually:
                stance_magnitude_thresholds = {"LF": None, "RF": None}
                stance_magnitude_threshold = 1.6
                for foot in ["RF", "LF"]:
                    imu_path = os.path.join(run_directory, "IMU", foot + ".csv")
                    imu = IMU(imu_path)
                    imu.gyro_to_rad()

                    stance_count_threshold = 8
                    stance = gyro_threshold_stance(
                        imu,
                        stance_magnitude_threshold=stance_magnitude_threshold,
                        stance_count_threshold=stance_count_threshold,
                    )

                    fig = plot_gyro_magnitude(
                        imu, stance=stance, threshold=stance_magnitude_threshold
                    )

                    fig.canvas.mpl_connect("key_press_event", stance_onclick)

                    show()

                    stance_magnitude_thresholds[foot] = stance_magnitude_threshold

                gyro_thresholds.append(
                    [
                        subject,
                        run,
                        stance_magnitude_thresholds["LF"],
                        stance_magnitude_thresholds["RF"],
                    ]
                )

    if check_stance_phase_manually:
        pd.DataFrame(
            data=gyro_thresholds,
            columns=[
                "subject",
                "run",
                "stance_magnitude_threshold_left",
                "stance_magnitude_threshold_right",
            ],
        ).to_csv(
            os.path.join(base_path, "gyro_stance_magnitude_thresholds.csv"), index=False
        )

    if check_timestamps:
        pd.DataFrame(data=ic_timestamps, columns=["subject", "run", "ic_time"]).to_csv(
            os.path.join(base_path, "ic_timestamps.csv"), index=False
        )
