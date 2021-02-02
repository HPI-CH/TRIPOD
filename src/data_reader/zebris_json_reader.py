""" This module contains the ZebrisJsonReader. """

import gzip
import json
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from PIL import Image

from matplotlib import cm

from scipy.ndimage import measurements


class ZebrisJsonReader:
    """
    The ZebrisJsonReader can be used to read and load preprocessed Zebris files.
    Besides the functionality for extracting gait parameters from the data,
    this class provides functionality for creating images and videos from the data.
    """

    def __init__(self, raw_data_file, step_data_file):
        """
        Initialization of the ZebrisJsonReader.

        Args:
            raw_data_file (str): raw data file name
            step_data_file (str): step data file name
        """
        self.raw_data_file = raw_data_file
        self.step_data_file = step_data_file
        self.raw_data = None
        self.step_data = None

    def get_raw_data(self):
        """
        Get raw data.
        Lazy loading with caching.

        Returns:
            dict: raw data
        """
        if not self.raw_data:
            self.raw_data = json.load(gzip.open(self.raw_data_file, "r"))
        return self.raw_data

    def get_step_data(self):
        """
        Get step (aggregated) data.
        Lazy loading with caching.

        Returns:
            dict: step data
        """
        if not self.step_data:
            self.step_data = json.load(gzip.open(self.step_data_file, "r"))
        return self.step_data

    def read_treadmill_velocity(self):
        """
        Read treadmill velocity.

        Returns:
            list[float]: Treadmill velocity for each sample.
        """
        data = self.get_raw_data()

        treadmill_velocity = [sample["velocity"] for sample in data["samples"]]

        return treadmill_velocity

    def read_zebris_raw_json_to_video(self, out_file):
        """
        Creates an mp4 video of the raw sensor data.
        The framerate will be the actual sensor reading rate (some players might have problems with unusual framerates).
        One pixel is one sensor cell.

        Args:
            out_file (str): mp4 file name

        Returns:
            None
        """

        data = self.get_raw_data()

        x_range = data["cell_count"]["x"]
        y_range = data["cell_count"]["y"]

        pressure = []
        for sample in data["samples"]:
            for value in sample["pressure"]:
                pressure.append(value)
        max_pressure = max(pressure)

        frequency = data["frequency"]

        color_map = cm.get_cmap("nipy_spectral")

        # this folder is created and deleted in the working directory
        # make sure, it doesn't already exists
        tmp_folder = "tmp_video_frames"
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)

        for sample_id, sample in enumerate(data["samples"]):
            image_data = np.zeros((x_range, y_range, 3))

            if len(sample["pressure"]):
                x_begin = sample["origin"]["x"]
                y_begin = sample["origin"]["y"]
                x_count = sample["size"]["x"]
                y_count = sample["size"]["y"]

                # read data FORTRAN-style (first index changes fastest)
                # flip it. Zebris y axis is pointing up
                cells = np.flip(
                    np.reshape(sample["pressure"], (x_count, y_count), order="F"),
                    axis=1,
                )
                cells = cells / max_pressure
                cells = color_map(cells)[:, :, :3] * 255

                image_data[
                    x_begin : x_begin + x_count, y_begin : y_begin + y_count, :
                ] = cells

            img = Image.fromarray(np.rot90(image_data).astype("uint8"), "RGB")
            img.save(os.path.join(tmp_folder, str(sample_id).zfill(6) + ".png"))

        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                str(frequency),
                "-i",
                tmp_folder + "/%06d.png",
                "-c:v",
                "copy",
                out_file,
            ],
            shell=True,
        )
        shutil.rmtree(tmp_folder)

    def read_zebris_raw_json_heel_positions(  # noqa: C901
        self, generate_images=False, generate_heel_image=False
    ):
        """
        Zebris does not provide stride length out of the box.
        Therefore, the positions of heel strikes are measured
        in order to obtain stride length.
        This function can also be used to create a series of images
        that could be later merged into a video or used to inspect the
        raw data. Also an image of the performed heel detection can
        be exported.

        Args:
            generate_images (bool): flag, if a series of images should be created
            generate_heel_image (bool): flag, if an image of the heel detection should be created

        Returns:
            (tuple[list[float], ...]): Positions of heelstrike for the left and right foot.
        """
        data = self.get_raw_data()

        x_range = data["cell_count"]["x"]
        y_range = data["cell_count"]["y"]

        cell_size = data["cell_size"]["y"]  # along the direction of walking

        treadmill_velocity = [sample["velocity"] for sample in data["samples"]]

        frequency = data["frequency"]

        total_distance = (
            1000 * np.sum(np.divide(treadmill_velocity, frequency)) / cell_size
        )
        total_distance = round(total_distance).astype("int") + y_range
        # total_distance = 800 # for faster debugging
        image_data = np.zeros((x_range, total_distance, 3))

        if generate_images or generate_heel_image:
            tmp_folder = "tmp_video_frames"
            if not os.path.exists(tmp_folder):
                os.mkdir(tmp_folder)

        if generate_images:
            pressure = []
            for sample in data["samples"]:
                for value in sample["pressure"]:
                    pressure.append(value)
            max_pressure = max(pressure)

            color_map = cm.get_cmap("nipy_spectral")

        for sample_id, sample in enumerate(data["samples"]):
            if len(sample["pressure"]):
                x_begin = sample["origin"]["x"]
                y_begin = sample["origin"]["y"]
                x_count = sample["size"]["x"]
                y_count = sample["size"]["y"]

                distance = (
                    1000
                    * np.sum(np.divide(treadmill_velocity[: sample_id + 1], frequency))
                    / cell_size
                )
                distance = round(distance).astype("int")

                cells = np.flip(
                    np.reshape(sample["pressure"], (x_count, y_count), order="F"),
                    axis=1,
                )

                if generate_images:
                    # convert previous image to grayscale
                    if np.any(image_data):
                        image_data[np.any(image_data, axis=2)] = 0x80

                    # convert data to color
                    cells = cells / max_pressure
                    cells = color_map(cells)[:, :, :3] * 255

                    # insert current footprint into previous image
                    try:  # necessary if we don't want the full image
                        image_data[
                            x_begin : x_begin + x_count,
                            distance + y_begin : distance + y_begin + y_count,
                            :,
                        ][np.any(cells, axis=2)] = cells[np.any(cells, axis=2)]
                    except IndexError:
                        break

                    # save image
                    img = Image.fromarray(image_data.astype("uint8"), "RGB")
                    img.save(os.path.join(tmp_folder, str(sample_id).zfill(6) + ".png"))

                else:
                    try:  # necessary if we don't want the full image
                        image_data[
                            x_begin : x_begin + x_count,
                            distance + y_begin : distance + y_begin + y_count,
                            :,
                        ][cells != 0] = 0x80
                    except IndexError:
                        break
            else:
                if generate_images:
                    # convert previous image to grayscale
                    if np.any(image_data):
                        image_data[np.any(image_data, axis=2)] = 0x80

                    # save image
                    img = Image.fromarray(image_data.astype("uint8"), "RGB")
                    img.save(os.path.join(tmp_folder, str(sample_id).zfill(6) + ".png"))

        # find all clusters
        labels, components = measurements.label(np.any(image_data, axis=2))
        heel_y = []

        # get the minimal y-coordinate of each cluster
        for component in range(1, components + 1):
            if np.count_nonzero(labels == component) > 50:
                heel_y.append(np.min(np.nonzero(labels == component)[1]))

        # clusters may be unordered. order them by y position
        heel_y = np.sort(heel_y)

        # a footprint may consist of multiple clusters (e.g. heel and ball of the foot)
        # therefore "merge" clusters that are closer than max_cluster_size in cm away from eachother
        # max_cluster_size is chosen to be greater than the longest footlength and shorter than the smallest step length
        max_cluster_size = 40
        filtered_heel_y = [heel_y[0]]
        current_y = heel_y[0]
        for y in heel_y[1:]:
            if np.abs(y - current_y) > (max_cluster_size * (cell_size / 10)):
                filtered_heel_y.append(y)
                current_y = y
        heel_y = np.array(filtered_heel_y)

        # assume that the subject start with the right foot and seperate heel positions into right and left foot
        heel_y_right = heel_y[
            (np.arange(0, np.floor(len(heel_y) / 2)) * 2).astype(int)
        ].astype(int)
        heel_y_left = heel_y[
            (np.arange(0, np.floor(len(heel_y) / 2)) * 2 + 1).astype(int)
        ].astype(int)

        # get the mean position of heelstrike aloong the x-axis ...
        right_x_mean = np.mean(
            [
                np.mean(np.nonzero(image_data[:, heel_pos, 1])[0])
                for heel_pos in heel_y_right
            ]
        )
        left_x_mean = np.mean(
            [
                np.mean(np.nonzero(image_data[:, heel_pos, 1])[0])
                for heel_pos in heel_y_left
            ]
        )

        assert right_x_mean > left_x_mean

        # and correct assumption about starting foot
        # if right_x_mean < left_x_mean:
        #    print("switching starting foot")
        #    heel_y_right, heel_y_left = heel_y_left, heel_y_right

        if generate_heel_image:
            for heel_pos in heel_y_left:
                image_data[:, int(heel_pos), :] = [0, 255, 0]
            for heel_pos in heel_y_right:
                image_data[:, int(heel_pos), :] = [0, 0, 255]

            img = Image.fromarray(image_data.astype("uint8"), "RGB")
            img.save(os.path.join(tmp_folder, "steps.png"))

        # convert to meters
        heel_y_right_m = heel_y_right * cell_size / 1000
        heel_y_left_m = heel_y_left * cell_size / 1000

        return heel_y_left_m, heel_y_right_m

    def read_zebris_raw_json_ic_fo(self):
        """
        Read the initial contact and foot-off timestamps.
        Zebris provides this data as part of the aggregated data.

        Returns:
            tuple[DataFrame, ...]: DataFrames of IC and FO timestamps for the left and right foot.
        """

        data = self.get_step_data()

        right_ic_fo = {"IC": [], "FO": []}
        left_ic_fo = {"IC": [], "FO": []}

        for event in data["events"]:
            begin = event["begin"]
            end = event["end"]
            side = event["side"]

            if side == "left":
                ic_fo = left_ic_fo
            else:
                ic_fo = right_ic_fo
            ic_fo["IC"].append(begin)
            ic_fo["FO"].append(end)

        return pd.DataFrame(left_ic_fo), pd.DataFrame(right_ic_fo)

    def read_zebris_raw_json_initial_contact(self):
        """
        The aggregated data from Zebris does not contain the first step.
        However, the raw data does. Therefore, the initial contact timestamp
        needs to be extracted from the raw data and is not the start of the first
        "detected step".

        Returns:
            float: IC timestamp
        """

        # use zebris_raw
        data = self.get_raw_data()

        frequency = data["frequency"]

        ic_sample = 0
        for sample_id, sample in enumerate(data["samples"]):
            if len(sample["pressure"]):
                ic_sample = sample_id
                break

        ic_time = data["begin"] + ic_sample / frequency

        return ic_time


if __name__ == "__main__":

    raw_file = "./example_data/raw/TRIPOD/Sub_FZ/PWS/Zebris/zebris_raw.json.gz"
    steps_file = "./example_data/raw/TRIPOD/Sub_FZ/PWS/Zebris/zebris_steps.json.gz"

    reader = ZebrisJsonReader(raw_file, steps_file)

    reader.read_zebris_raw_json_to_video("./FZ_PWS_zebris.mp4")

    # print(reader.read_zebris_raw_json_initial_contact())

    # reader.read_zebris_raw_json_heel_positions(False, True)

    # print(reader.read_zebris_raw_json_ic_fo())
