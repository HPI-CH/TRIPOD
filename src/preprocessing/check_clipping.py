""" This script checks if the recorded data is clipping.
    This can happen, since the IMUs record in a fixed range.
"""

from src.data_reader.imu import IMU
import matplotlib.pyplot as plt
import os
import fnmatch
import numpy as np

if __name__ == "__main__":

    # Check for all csv files in the base directory if the signal exceeds the minimum or maximum threshold

    base_path = "./data/raw/TRIPOD/"

    for root, d_names, f_names in os.walk(base_path):
        for f in f_names:
            # only check feet (instep) signals
            # however, heel signals are more likely to clip
            if fnmatch.fnmatch(f, "?F.csv"):
                file_name = os.path.join(root, f)

                imu = IMU(file_name)

                # adapt range depending on chosen range
                range_max = 7.9

                t = imu.time()
                accel = np.transpose(imu.accel())

                if (
                    len(
                        np.nonzero(
                            np.logical_or(
                                accel > range_max, accel < -range_max
                            ).flatten()
                        )[0]
                    )
                    > 4
                ):
                    print(file_name)
                    plt.figure()
                    for dim in range(0, 3):
                        plt.plot(t, accel[dim])
                        plt.scatter(
                            t[accel[dim] > range_max],
                            accel[dim][accel[dim] > range_max],
                        )
                        plt.scatter(
                            t[accel[dim] < -range_max],
                            accel[dim][accel[dim] < -range_max],
                        )

                    plt.show()
