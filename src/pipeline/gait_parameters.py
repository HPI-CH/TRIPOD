""" This module contains the GaitParameters class. """

import numpy as np
import pandas as pd

# A gait cycle is defined to start with stance phase, followed by swing phase
# Thus a proper recording has an initial contact event as first and last event


class GaitParameters:
    """
    This class merges trajectories and gait events together and calculates
    the gait parameters stride length, stride time, swing time and stance time.
    """

    def __init__(self, trajectories, gait_events, initial_contact):
        """
        Initialization of GaitParameters

        Args:

            trajectories (dict[str, DataFrame]): DataFrames with trajectory information for the left and right foot
            gait_events (dict[str, Any]): Gait events and definition of start and end event for the right and left foot
            initial_contact (float): Initial contact timestamp
        Returns:
            None
        """
        self.initial_contact = initial_contact
        self.trajectories = trajectories
        self.gait_events = gait_events
        self.adjust_data()

    def adjust_data(self):
        """
        Adjust gait events so that the data starts and ends with a start event.

        Returns:
            None
        """
        stance_begin = self.gait_events["stance_begin"]
        stance_end = self.gait_events["stance_end"]

        for side in ["left", "right"]:
            assert len(self.gait_events[side]["times"][stance_begin]) == len(
                self.gait_events[side]["times"][stance_end]
            )

            # if the recording doesn't start with start event
            while (
                self.gait_events[side]["samples"][stance_begin][0]
                >= self.gait_events[side]["samples"][stance_end][0]
            ):
                # drop first end event
                self.gait_events[side]["samples"][stance_end] = self.gait_events[side][
                    "samples"
                ][stance_end][1:]
                self.gait_events[side]["times"][stance_end] = self.gait_events[side][
                    "times"
                ][stance_end][1:]

            # if the recording doesn't end with start event
            if (
                self.gait_events[side]["samples"][stance_begin][-1]
                < self.gait_events[side]["samples"][stance_end][-1]
            ):
                # drop last end event
                self.gait_events[side]["samples"][stance_end] = self.gait_events[side][
                    "samples"
                ][stance_end][:-1]
                self.gait_events[side]["times"][stance_end] = self.gait_events[side][
                    "times"
                ][stance_end][:-1]

            # now there should be exactly one more start event than end event
            # print(len(self.gait_events[side]["times"][stance_begin]))
            # print(len(self.gait_events[side]["times"][stance_end]))
            assert (
                len(self.gait_events[side]["times"][stance_begin])
                == len(self.gait_events[side]["times"][stance_end]) + 1
            )

    def summary(self):
        """
        Calculate the actual gait parameters.

        Returns:
            DataFrame: DataFrame with the gait parameters
        """
        stride_length = self.stride_length()
        stride_time = self.stride_time()
        swing_time = self.swing_time()
        stance_time = self.stance_time()

        summary = {"left": None, "right": None}
        stance_begin = self.gait_events["stance_begin"]
        for side in summary.keys():
            summary[side] = pd.DataFrame(
                data={
                    "timestamp": np.array(
                        self.gait_events[side]["times"][stance_begin][:-1]
                    )
                    - self.initial_contact,
                    "stride_length": stride_length[side],
                    "stride_time": stride_time[side],
                    "swing_time": swing_time[side],
                    "stance_time": stance_time[side],
                }
            )
        return summary

    def stride_length(self):
        """
        Calculate stride length.

        Returns:
            dict[str, list[float]]: List of stride length for the right and left foot
        """
        stride_length = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        for side in stride_length.keys():
            for start, end in zip(
                self.gait_events[side]["samples"][stance_begin][:-1],
                self.gait_events[side]["samples"][stance_begin][1:],
            ):
                step = np.array(
                    [
                        self.trajectories[side]["position_x"][end],
                        self.trajectories[side]["position_y"][end],
                        self.trajectories[side]["position_z"][end],
                    ]
                ) - np.array(
                    [
                        self.trajectories[side]["position_x"][start],
                        self.trajectories[side]["position_y"][start],
                        self.trajectories[side]["position_z"][start],
                    ]
                )
                stride_length[side].append(np.linalg.norm(step[0:2]))

        return stride_length

    def stride_time(self):
        """
        Calculate stride time.

        Returns:
            dict[str, list[float]]: List of stride times for the right and left foot
        """
        stride_time = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        for side in stride_time.keys():
            stride_time[side] = np.array(
                self.gait_events[side]["times"][stance_begin][1:]
            ) - np.array(self.gait_events[side]["times"][stance_begin][:-1])
        return stride_time

    def swing_time(self):
        """
        Calculate swing time.

        Returns:
            dict[str, list[float]]: List of swing times for the right and left foot
        """
        swing_time = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        for side in swing_time.keys():
            stance_end = self.gait_events["stance_end"]
            swing_time[side] = np.array(
                self.gait_events[side]["times"][stance_begin][1:]
            ) - np.array(self.gait_events[side]["times"][stance_end])
        return swing_time

    def stance_time(self):
        """
        Calculate stance time.

        Returns:
            dict[str, list[float]]: List of stance times for the right and left foot
        """
        stance_time = {"left": [], "right": []}
        stance_begin = self.gait_events["stance_begin"]
        stance_end = self.gait_events["stance_end"]
        for side in stance_time.keys():
            stance_time[side] = np.array(
                self.gait_events[side]["times"][stance_end]
            ) - np.array(self.gait_events[side]["times"][stance_begin][:-1])
        return stance_time
