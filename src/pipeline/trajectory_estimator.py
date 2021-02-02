""" This module contains the implementation of a trajectory estimator. """

import pandas as pd
import os

from pipeline.abstract_pipeline_components import AbstractTrajectoryEstimator
from trajectory_estimation.filter import error_state_kalman_filter


class TuncaTrajectoryEstimator(AbstractTrajectoryEstimator):
    """
    Trajectory estimator based on Tunca et al. (https://doi.org/10.3390/s17040825).
    The actual error-state Kalman filter is implemented in trajectory_estimation/filter.py
    """

    def estimate(
        self,
        name,
        interim_base_path,
        dataset,
        subject,
        run,
        imu_ic,
        stance_thresholds,
        overwrite,
    ):
        """
        Estimate trajectories from IMU data.

        Args:
            name (str): Name of the configuration (used to identify cache files)
            interim_base_path (str): Folder where caching data can be stored
            dataset (str): Folder containing the dataset
            subject (str): Identifier of the subject
            run (str): Identifier of the run
            imu_ic (float): timestamp of initial contact in the IMU data
            stance_thresholds (dict[str, float]): Gyroscope magnitude and stance count thresholds for stance detection for the right and left foot
            overwrite (bool): Flag if cached files should be overwritten

        Returns:
            dict[str, DataFrame]: DataFrames with trajectory information for the right and left foot
        """

        interim_data_path = os.path.join(interim_base_path, dataset, subject, run)

        # check if cached results are present
        if (
            os.path.exists(
                os.path.join(interim_data_path, name + "_tunca_estimation_left.json")
            )
            and os.path.exists(
                os.path.join(interim_data_path, name + "_tunca_estimation_right.json")
            )
            and not overwrite
        ):
            return {
                "left": pd.read_json(
                    os.path.join(
                        interim_data_path, name + "_tunca_estimation_left.json"
                    )
                ),
                "right": pd.read_json(
                    os.path.join(
                        interim_data_path, name + "_tunca_estimation_right.json"
                    )
                ),
            }
        else:
            # calculate trajectories
            os.makedirs(interim_data_path, exist_ok=True)
            trajectories = {}
            for foot in [("left", "LF"), ("right", "RF")]:
                # "LF" and "RF" correspond to the filenames of the respective sensors
                trajectory = error_state_kalman_filter(
                    self.imus[foot[1]],
                    imu_ic,
                    zero_z=True,
                    zero_xz=False,
                    stance_magnitude_threshold=float(
                        stance_thresholds["stance_magnitude_threshold_" + foot[0]]
                    ),
                    stance_count_threshold=int(
                        stance_thresholds["stance_count_threshold_" + foot[0]]
                    ),
                )
                # cache results
                trajectory.to_json(
                    os.path.join(
                        interim_data_path,
                        name + "_tunca_estimation_" + foot[0] + ".json",
                    )
                )

                trajectories[foot[0]] = trajectory

            return trajectories
