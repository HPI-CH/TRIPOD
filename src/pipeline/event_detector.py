""" This module contains implementation of different gait event detectors. """

from pipeline.abstract_pipeline_components import AbstractEventDetector
from event_detection.imu_event_detection import (
    hundza_gait_events,
    tunca_gait_events,
)
import pandas as pd


class TuncaEventDetector(AbstractEventDetector):
    """
    Gait event detection as presented by Tunca et al. (https://doi.org/10.3390/s17040825).
    The actual algorithm is implemented in event_datection/imu_event_detection.py
    """

    def detect(self, stance_thresholds):
        """
        Detect gait events.

        Args:
            stance_thresholds (dict[str, float]): Gyroscope magnitude and stance count thresholds for stance detection

        Returns:
            dict[str, dict]: IC and FO samples and timestamps for the right and left foot.
        """
        result = {}

        result["stance_begin"] = "IC"
        result["stance_end"] = "FO"

        for foot in [("left", "LF"), ("right", "RF")]:
            IC_samples, FO_samples, IC_times, FO_times, stance = tunca_gait_events(
                self.imus[foot[1]],
                float(stance_thresholds["stance_magnitude_threshold_" + foot[0]]),
                int(stance_thresholds["stance_count_threshold_" + foot[0]]),
            )

            result[foot[0]] = {
                "samples": {"IC": IC_samples, "FO": FO_samples},
                "times": {"IC": IC_times, "FO": FO_times},
            }

        return result


class HundzaEventDetector(AbstractEventDetector):
    """
    Gait event detection by Hundza et al. (https://doi.org/10.1109/TNSRE.2013.2282080)
    An example of another gait event detection algorithm.
    This EventDetector is for demonstration only and has not been evaluated.
    """

    def detect(self):
        """
        Detect gait events.

        Returns:
            dict[str, dict]: TOFS, IOFS and TO events
        """
        result = {}

        for foot in [("right", "RL"), ("left", "LL")]:
            TOFS, IOFS, TO, stance = hundza_gait_events(self.imus[foot[1]])
            result[foot[0]] = pd.DataFrame(data={"TOFS": TOFS, "IOFS": IOFS, "TO": TO})

        return result
