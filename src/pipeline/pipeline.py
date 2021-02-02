"""This module contains the pipeline class."""

import os
import numpy as np
import pandas as pd

# all non-variable parts of the pipeline need to be imported
from pipeline.gait_parameters import GaitParameters
from pipeline.evaluator import Evaluator


class Pipeline:
    """
    This class is the skeleton for all possible pipeline instantiations.

    It is instantiated with a config dictionary that defines the different variable components
    of the pipeline, the data source and auxillary variables.
    """

    def __init__(self, pipeline_config):
        """Instantiation of a pipeline.
        For the necessary variables and explanation of the configuration dictionary see pipeline_playground.py.

        Args:
            pipeline_config (dict): Dictionary containing configuration variables
        """
        self.config = pipeline_config
        self.imu_ic = None
        self.imu_gyro_threshold = None
        self.evaluator = Evaluator()

        # load initial contacts from csv file
        self.imu_ic_timestamps = pd.read_csv(
            os.path.join(
                self.config["raw_base_path"],
                self.config["dataset"],
                "SyncInfo.csv",
            )
        )

        # load gyro thresholds from csv file
        self.imu_gyro_thresholds = pd.read_csv(
            os.path.join(
                self.config["raw_base_path"],
                self.config["dataset"],
                "stance_magnitude_thresholds_manual.csv",
            )
        )

    def load_data(self, subject_num, run_num):
        """
        Load all IMU data for the given subject_num and run_num from the IMU folder.

        Args:
            subject_num (int): Index of the subject whose data should be loaded

            run_num (int): Index of the run/trial that should be loaded

        Returns:
            tuple[dict[str, IMU], float): Tuple of a dictionary of IMU objects and the timestamp of initial contact in seconds
        """
        imus = self.config["data_loader"](
            self.config["raw_base_path"],
            self.config["dataset"],
            self.config["subjects"][subject_num],
            self.config["runs"][run_num],
        ).get_data()

        imu_ic = float(
            self.imu_ic_timestamps[
                np.logical_and(
                    self.imu_ic_timestamps["subject"]
                    == self.config["subjects"][subject_num],
                    self.imu_ic_timestamps["run"] == self.config["runs"][run_num],
                )
            ]["ic_time"]
        )

        # crop imu data to fit experiment_duration seconds from inititial contact
        # start 2 seconds earlyer to get also lift-off data before initial contact
        for imu in imus.values():
            imu.crop(
                imu_ic - 2,
                imu_ic + self.config["experiment_duration"],
                inplace=True,
            )

        self.stance_thresholds = self.imu_gyro_thresholds[
            np.logical_and(
                self.imu_gyro_thresholds["subject"]
                == self.config["subjects"][subject_num],
                self.imu_gyro_thresholds["run"] == self.config["runs"][run_num],
            )
        ][
            [
                "stance_magnitude_threshold_left",
                "stance_magnitude_threshold_right",
                "stance_count_threshold_left",
                "stance_count_threshold_right",
            ]
        ]

        return imus, imu_ic

    def detect_gait_events(self, imus):
        """Perform gait event detection with the gait event detector specified in the config.

        Args:
            imus (dict[str, IMU]): IMU objects for each sensor location

        Returns:
            dict[str, dict]: IC and FO samples and timestamps for the right and left foot
        """
        return self.config["gait_event_detector"](imus).detect(
            self.stance_thresholds,
        )

    def estimate_trajectories(self, subject_num, run_num, imus, imu_ic):
        """
        Estimate left and right foot trajectories using the specified trajectory estimation algorithm.
        Subject and run need to be identified since the trajectory estimator uses caching.

        Args:
            subject_num (int): subject index
            run_num (int): run index
            imus (dict[str, IMU]):  IMU objects for each sensor location
            imu_ic (float): Inital contact timestamp

        Returns:
            dict[str, DataFrame]: DataFrames with trajectory information for the right and left foot
        """
        return self.config["trajectory_estimator"](imus).estimate(
            self.config["name"],
            self.config["interim_base_path"],
            self.config["dataset"],
            self.config["subjects"][subject_num],
            self.config["runs"][run_num],
            imu_ic,
            self.stance_thresholds,
            self.config["overwrite"],
        )

    def calculate_gait_parameters(self, gait_events, trajectories, imu_ic):
        """Calculate gait parameters.

        Args:
            gait_events (dict[str, dict]): IC and FO samples and timestamps for the right and left foot
            trajectories (dict[str, DataFrame]): DataFrames with trajectory information for the right and left foot
            imu_ic (float): Inital contact timestamp

        Returns:
            dict[str, DataFrame]: DataFrame with the estimated gait parameters for the left and right foot
        """

        return GaitParameters(trajectories, gait_events, imu_ic).summary()

    def load_reference_data(self, subject_num, run_num):
        """Load reference data with the specified reference loader.
        Note: subject and run need to be identified since the reference loader uses caching.

        Args:
            subject_num (int): subject index
            run_num (int): run index

        Returns:
            dict[str, DataFrame]: DataFrames with gait parameters for the left and right foot
        """

        return self.config["reference_loader"](
            self.config["name"],
            self.config["raw_base_path"],
            self.config["interim_base_path"],
            self.config["dataset"],
            self.config["subjects"][subject_num],
            self.config["runs"][run_num],
            self.config["overwrite"],
        ).get_data()

    def add_to_evaluator(self, subject_num, run_num, reference_data, gait_parameters):
        """Add estimated gait parameters and reference data for one subject and run to the evaluator.

        Args:
            subject_num (int): subject index
            run_num (int): run index
            reference_data (dict[str, DataFrame]): DataFrames with reference gait parameters for the left and right foot
            gait_parameters (dict[str, DataFrame]): DataFrame with the estimated gait parameters for the left and right foot

        Returns:
            None
        """

        self.evaluator.add_data(subject_num, run_num, gait_parameters, reference_data)

    def execute(self, subject_runs):
        """
        Core function of the pipeline.
        For each subject and run:
        Load IMU data, estimate trajectories, detect gait events,
        calculate estimated gait parameters, add them together with the
        reference_data to the evaluator.
        Exectue the evaluator with the results of all runs altogether.

        Args:
            subject_runs (tuple[int, int]): Index of the subject and run whose data should be loaded

        Returns:
            None
        """
        # executes all pipeline stages except the evaluator for all specified (subject_id, run_id) tuple
        for subject_num, run_num in subject_runs:
            print(
                "processing subject",
                self.config["subjects"][subject_num],
                "run",
                self.config["runs"][run_num],
            )

            print("load data")
            imu_data, imu_ic = self.load_data(subject_num, run_num)

            print("load reference data")
            reference_data = self.load_reference_data(subject_num, run_num)

            print("detect gait events")
            gait_events = self.detect_gait_events(imu_data)

            print("estimate trajectories")
            trajectories = self.estimate_trajectories(
                subject_num, run_num, imu_data, imu_ic
            )

            print("calculate gait parameters")
            gait_parameters = self.calculate_gait_parameters(
                gait_events, trajectories, imu_ic
            )

            self.add_to_evaluator(subject_num, run_num, reference_data, gait_parameters)

        # match reference system and estimated gait parameters stride by stride
        self.evaluator.match_timestamps()

        # generate plots
        self.evaluator.plot_correlation(
            "Tunca et al.", "stride_length", subject_runs, self.config["reference_name"]
        )
        self.evaluator.plot_correlation(
            "Tunca et al.", "stride_time", subject_runs, self.config["reference_name"]
        )
        self.evaluator.plot_bland_altmann(
            "stride_length", subject_runs, self.config["reference_name"]
        )
        self.evaluator.plot_bland_altmann(
            "stride_time", subject_runs, self.config["reference_name"]
        )
        # self.evaluator.plot_correlation(
        #    "Tunca et al.", "stance_time", subject_runs, self.config["reference_name"]
        # )
        # self.evaluator.plot_correlation(
        #    "Tunca et al.", "swing_time", subject_runs, self.config["reference_name"]
        # )
