"""
This module contains the implementations of actual IMU data loaders
A data loader is expected to implement functionality that builds a
number of IMU objects from data specified by the given parameters.
"""

import fnmatch
import os
import pandas as pd

from data_reader.imu import IMU
from pipeline.abstract_pipeline_components import AbstractDataLoader


class PhysilogDataLoader(AbstractDataLoader):
    """
    This class reads data from CSV files as generated by the PhysilogRTK.
    """

    def load(self, raw_base_path, dataset, subject, run):
        """
        The load function is called at instantiation (see AbstractDataLoader).

        It loads the data specified by the given parameters and stores it into self.data.

        Args:
            raw_base_path (str): Base folder for every dataset
            dataset (str): Folder containing the dataset
            subject (str): Subject identifier
            run (str): Run identifier

        Returns:
            None

        """
        # construct path to actual data files
        data_path = os.path.join(
            raw_base_path,
            dataset,
            subject,
            run,
            "IMU",
        )

        for file_name in os.listdir(data_path):
            if fnmatch.fnmatch(file_name, "*.csv"):
                csv_file = os.path.join(data_path, file_name)
                imu_data = pd.read_csv(csv_file, header=4, skiprows=[6])
                imu_data = imu_data[
                    [
                        "Time",
                        "Gyro X",
                        "Gyro Y",
                        "Gyro Z",
                        "Accel X",
                        "Accel Y",
                        "Accel Z",
                    ]
                ]
                imu = IMU(imu_data)
                imu.acc_to_meter_per_square_sec()
                imu.gyro_to_rad()
                self.data[os.path.splitext(file_name)[0]] = imu


class MatlabDataLoader(AbstractDataLoader):
    """
    This class reads data from a CSV file as exported from a MATLAB script.
    """

    def load(self, raw_base_path, dataset, subject, run):
        """
        The load function is called at instantiation (see AbstractDataLoader).

        It loads the data specified by the given parameters and stores it into self.data.

        Args:
            raw_base_path (str): Base folder for every dataset
            dataset (str): Folder containing the dataset
            subject (str): Subject identifier
            run (str): Run identifier

        Returns:
            None

        """
        data_path = os.path.join(
            raw_base_path,
            dataset,
            subject,
            run,
            "IMU",
        )

        for file_name in os.listdir(data_path):
            if fnmatch.fnmatch(file_name, "*.csv"):
                csv_file = os.path.join(data_path, file_name)
                imu_data = pd.read_csv(csv_file)
                imu_data = imu_data[
                    [
                        "Time",
                        "Gyro X",
                        "Gyro Y",
                        "Gyro Z",
                        "Accel X",
                        "Accel Y",
                        "Accel Z",
                    ]
                ]
                imu = IMU(imu_data)
                self.data[os.path.splitext(file_name)[0]] = imu