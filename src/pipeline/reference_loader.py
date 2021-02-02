"""This module contains the data loaders for different reference systems."""

import os
import fnmatch
import pandas as pd

from pipeline.abstract_pipeline_components import AbstractReferenceLoader
from src.data_reader.zebris_json_reader import ZebrisJsonReader


class ZebrisReferenceLoader(AbstractReferenceLoader):
    """
    This class loads reference data created by the Zebris FDM-THQ system.
    It uses caching, since the extraction of foot positions from the Zebris raw data
    requires some computation and only needs to be done once.
    This computation is outsourced to the ZebrisJsonReader that can be also used to generate
    visualizations and inspect the data.
    """

    def load(self):
        """
        Load the data based on parameters provided in the constructor.

        Returns:
            None
        """
        # construct interim path
        self.interim_data_path = os.path.join(
            self.interim_base_path, self.dataset, self.subject, self.run
        )

        # check if cached data is present
        if (
            os.path.exists(
                os.path.join(self.interim_data_path, self.name + "_zebris_left.json")
            )
            and os.path.exists(
                os.path.join(self.interim_data_path, self.name + "_zebris_right.json")
            )
            and not self.overwrite
        ):
            self.load_interim_data()
        else:
            self.raw_data_path = os.path.join(
                self.raw_base_path, self.dataset, self.subject, self.run, "Zebris"
            )

            os.makedirs(self.interim_data_path, exist_ok=True)
            self.load_raw_data()

    def load_interim_data(self):
        """
        Load data from cached files.

        Returns:
            None
        """
        for side in self.data.keys():
            self.data[side] = pd.read_json(
                os.path.join(
                    self.interim_data_path, self.name + "_zebris_" + side + ".json"
                )
            )

    def load_raw_data(self):
        """
        Load data from raw data files.
        Zebris has two types of files (_raw and _steps).
        _raw files contain raw sensor readings.
        _steps files contain aggregated data per roll-off cycle.

        Returns:
            None
        """
        for file_name in os.listdir(self.raw_data_path):
            if fnmatch.fnmatch(file_name, "*raw.json.gz"):
                raw_json_file = os.path.join(self.raw_data_path, file_name)
            if fnmatch.fnmatch(file_name, "*steps.json.gz"):
                steps_json_file = os.path.join(self.raw_data_path, file_name)

        reader = ZebrisJsonReader(raw_json_file, steps_json_file)

        initial_contact = reader.read_zebris_raw_json_initial_contact()

        zebris_ic_fo = {}
        (
            zebris_ic_fo["left"],
            zebris_ic_fo["right"],
        ) = reader.read_zebris_raw_json_ic_fo()

        zebris_heel_positions = {}
        (
            zebris_heel_positions["left"],
            zebris_heel_positions["right"],
        ) = reader.read_zebris_raw_json_heel_positions()

        # clean up the data obtained from the zebris system
        for side in self.data.keys():
            heel_pos = zebris_heel_positions[side]

            # drop the first heel position since zebris doesn't track the time of the first rollover
            if side == "right":
                heel_pos = heel_pos[1:]

            # delete ic/fo pairs at the end
            min_steps = min(len(zebris_ic_fo[side]), len(heel_pos))
            zebris_ic_fo[side] = zebris_ic_fo[side][:min_steps]
            heel_pos = heel_pos[:min_steps]

            # make sure that the number of initial contact and foot off events matches up
            ic = zebris_ic_fo[side]["IC"].to_numpy()
            fo = zebris_ic_fo[side]["FO"].to_numpy()
            assert len(ic) == len(fo)

            # calculate actual gait parameters
            self.data[side] = pd.DataFrame(
                data={
                    "timestamp": ic[:-1] - initial_contact,
                    "stride_length_ref": heel_pos[1:] - heel_pos[:-1],
                    "stride_time_ref": ic[1:] - ic[:-1],
                    "swing_time_ref": ic[1:] - fo[:-1],
                    "stance_time_ref": fo[:-1] - ic[:-1],
                }
            )

            # save files for caching
            self.data[side].to_json(
                os.path.join(
                    self.interim_data_path, self.name + "_zebris_" + side + ".json"
                )
            )


class OptogaitReferenceLoader(AbstractReferenceLoader):
    """
    This class loads reference data created by the OptoGait system.
    """

    def load(self):
        """
        Load the data based on parameters provided in the constructor.

        Returns:
            None
        """
        self.raw_data_path = os.path.join(
            self.raw_base_path,
            self.dataset,
            self.subject,
            self.run,
            "OptoGait",
            "optogait.csv",
        )

        opto_gait_data = pd.read_csv(self.raw_data_path)

        # Explanation of Optogait column names as exported by the OptoGait software:
        # # : Step index
        # L/R : left or right foots
        # TStep : step time (time between initial contacts) in s
        # Step : step length in cm
        # Split : initial contact timestamp in s
        # Stride : stride length in cm
        # StrideTime\Cycle : stride time in s
        # TStance : stance time in s
        # TSwing : swing time in s

        # select only relevant columns
        opto_gait_data = opto_gait_data[
            ["L/R", "Split", "Stride", "StrideTime\\Cycle", "TStance", "TSwing"]
        ]
        # rename columns
        opto_gait_data.rename(
            columns={
                "Split": "timestamp",
                "Stride": "stride_length_ref",
                "StrideTime\\Cycle": "stride_time_ref",
                "TStance": "stance_time_ref",
                "TSwing": "swing_time_ref",
            },
            inplace=True,
        )
        # convert cm to m
        opto_gait_data.stride_length_ref = opto_gait_data.stride_length_ref / 100

        for side in [("left", "L"), ("right", "R")]:
            # seperate right and left foot data based on "L/R" column
            self.data[side[0]] = opto_gait_data[opto_gait_data["L/R"] == side[1]].loc[
                :, opto_gait_data.columns != "L/R"
            ]

            # since optogait stores data "by contact", thus by step and not by stride, some columns are shifted.
            self.data[side[0]].stride_length_ref = self.data[
                side[0]
            ].stride_length_ref.shift(-1)
            self.data[side[0]].stride_time_ref = self.data[
                side[0]
            ].stride_time_ref.shift(-1)
            self.data[side[0]].swing_time_ref = self.data[side[0]].swing_time_ref.shift(
                -1
            )

            # drop null values
            self.data[side[0]].dropna(inplace=True)

        # zero base timestamps to the initial contact (first contact of right foot)
        self.data["left"].timestamp -= self.data["right"].timestamp.iloc[0]
        self.data["right"].timestamp -= self.data["right"].timestamp.iloc[0]
