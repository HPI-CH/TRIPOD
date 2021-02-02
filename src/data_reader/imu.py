""" This module contains the IMU class. """

import numpy as np
import pandas as pd
import copy


class IMU:
    """
    An IMU object holds all the data recorded by an actual IMU and offers getter and conversion methods.
    """

    def __init__(self, data):
        """
        The imu object expects a pandas DataFrame with the following columns:
        Time, Gyro X, Gyro Y, Gyro Z, Accel X, Accel Y, Accel Z

        Args:
            data DataFrame: IMU data
        """

        self.data = data

    # Getter methods
    def time(self, i=None):
        """
        Getter method for timestamps.

        Depending on if i is specified, this returns either data from timestep i
        or the whole timeseries data.

        Args:
            i (int): Optional - index of the data

        Returns:
            float or np.array: data at the specified timestep i or all data
        """

        if i is None:
            return self.data["Time"].to_numpy()

        return self.data["Time"].to_numpy()[i]

    def accel(self, i=None):
        """
        Getter method for acceleration.

        Depending on if i is specified, this returns either data from timestep i
        or the whole timeseries data.

        Args:
            i (int): Optional - index of the data

        Returns:
            float or np.array: data at the specified timestep i or all data
        """
        if i is None:
            return self.data[["Accel X", "Accel Y", "Accel Z"]].to_numpy()

        return self.data[["Accel X", "Accel Y", "Accel Z"]].to_numpy()[i]

    def gyro(self, i=None):
        """
        Getter method for gyroscope data.

        Depending on if i is specified, this returns either data from timestep i
        or the whole timeseries data.

        Args:
            i (int): Optional - index of the data

        Returns:
            float or np.array: data at the specified timestep i or all data
        """
        if i is None:
            return self.data[["Gyro X", "Gyro Y", "Gyro Z"]].to_numpy()

        return self.data[["Gyro X", "Gyro Y", "Gyro Z"]].to_numpy()[i]

    def crop(self, min_time, max_time, inplace=False):
        """
        Crops data to timeframe between min_time and max_time.
        Attention: all other data is dropped.

        Args:
            min_time (float): start time of cropping
            max_time (float): end time of cropping
            inplace (bool): flag if cropping should be performed inplace or on a copy

        Returns:
            None or IMU: If not inplace, a cropped copy of the IMU object is returned.
        """
        if inplace:
            self.data = self.data[
                (self.data["Time"] >= min_time) & (self.data["Time"] < max_time)
            ]
            self.data.reset_index(drop=True, inplace=True)
        else:
            self_copy = copy.deepcopy(self)
            self_copy.data = self_copy.data[
                (self_copy.data["Time"] >= min_time)
                & (self_copy.data["Time"] < max_time)
            ]
            self_copy.data.reset_index(drop=True, inplace=True)
            return self_copy

    def gyro_to_rad(self):
        """Conversion method for gyroscope data from degree to rad."""
        self.data[["Gyro X", "Gyro Y", "Gyro Z"]] *= np.pi / 180

    def gyro_to_degree(self):
        """Conversion method for gyroscope data from rad to degree."""
        self.data[["Gyro X", "Gyro Y", "Gyro Z"]] /= np.pi / 180

    def acc_to_meter_per_square_sec(self):
        """Conversion method for acceleration data from g to m/s**2."""
        self.data[["Accel X", "Accel Y", "Accel Z"]] *= 9.80665

    def acc_to_g(self):
        """Conversion method for acceleration data from m/s**2 to g."""
        self.data[["Accel X", "Accel Y", "Accel Z"]] /= 9.80665

    # time manipulation methods
    # not actually used in this pipeline

    def zero_base_time(self):
        """Let timestamps start at zero."""
        self.data["Time"] -= self.data["Time"][0]

    def time_shift(self, time_offset):
        """Shift time by a given offset.

        Args:
            time_offset (float): offset used to shift time

        Returns:
            None
        """
        self.data["Time"] += time_offset

    def resample(self, f):
        """
        Resample a copy of IMU data to given frequency.

        Args:
            f (float): target frequency

        Returns:
            IMU: resampled IMU copy
        """
        self_copy = copy.deepcopy(self)
        self_copy.data = self_copy.data.set_index("Time")
        self_copy.data.index = pd.to_datetime(
            (self_copy.data.index.array * 1e9).astype("int64")
        )
        self_copy.data.index.name = "Time"
        self_copy.data = (
            self_copy.data.resample(pd.Timedelta(1.0 / f, unit="s"))
            .mean()
            .interpolate()
        )
        self_copy.data.index = (
            self_copy.data.index - pd.to_datetime(0)
        ).total_seconds()
        self_copy.data.reset_index(inplace=True)
        return self_copy

    # data inspection methods
    # not actually used in this pipeline

    def check_sampling(self):
        """Calculate mean and standard deviation of the sampling rate.

        Returns:
            tuple(float, float): mean and std. deviation of sampling rate
        """
        T = (np.append(self.time(), [0]) - np.append([0], self.time()))[1:-1]
        f = 1 / T
        mean = np.mean(f)
        stddev = np.std(f)
        # hist = plt.hist(f, bins='auto')
        # plt.show()
        return mean, stddev

    def accel_variance(self):
        """Calculate acceleration variance.

        Returns:
            float: variance of acceleration
        """
        return np.var(self.accel, axis=0)
