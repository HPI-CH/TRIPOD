""" This script is meant to demonstrate the usage of the pipeline.
    Please note inline documentation.
"""

# The general structure of this gait analysis pipeline looks as follows:
#
#                  ---- TrajectoryEstimator* ---
#                 /                             \
# DataLoader*----<                               >------ GaitParameters -----
#                 \                             /                            \
#                  ------ EventDetector*--------                              >--- Evaluator
#                                                                            /
# ReferenceLoader*-----------------------------------------------------------
#
# Stages marked with a * are modifiable and their interfaces are defined in pipeline/abstract_pipeline_components.py
# New derived classes may be implemented in order to use other data formats or algorithms

import sys,os
sys.path.append(os.getcwd())

# all variable pipeline components that are used in the pipeline need to be imported
from pipeline.data_loader import PhysilogDataLoader
from pipeline.event_detector import TuncaEventDetector
from pipeline.trajectory_estimator import TuncaTrajectoryEstimator
from pipeline.reference_loader import ZebrisReferenceLoader

from pipeline.reference_loader import OptogaitReferenceLoader

from pipeline.pipeline import Pipeline

if __name__ == "__main__":
    # configuration of the pipeline
    # the data processing pipeline is instantiated with the paramters specified here.
    pipeline_config = {
        # @name: the name should be unique for each pipeline configuration.
        # it is used to identify interim data and reuse it in the next run
        "name": "all_manual_optogait",
        # @raw_base_path: path to the folder containing the dataset.
        # "raw_base_path": "./data/raw",
        "raw_base_path": "./example_data/raw",
        # @interim_base_path: path to the folder where interim data can be stored.
        # "interim_base_path": "./data/interim",
        "interim_base_path": "./example_data/interim",
        # @overwrite: if True, all steps in the pipeline are executed again and interim files are overwritten
        # if Flase, interim files are used where present.
        "overwrite": False,
        # @dataset: name of the folder of the dataset within raw_base_path
        "dataset": "TRIPOD_excerpt",
        # @subjects: names of the subfolders in the dataset folder for each subject
        "subjects": [
            # "Sub_DF",
            # "Sub_HA",
            # "Sub_PB",
            # "Sub_AL",
            # "Sub_EN",
            # "Sub_KP",
            # "Sub_RW",
            # "Sub_BK",
            "Sub_FZ",
            # "Sub_LU",
            # "Sub_SN",
            # "Sub_CP",
            # "Sub_GK",
            # "Sub_OD",
            # "Sub_YU",
        ],
        # @subjects: names of the subfolders for each trial/run in the subject folders
        "runs": ["PWS", "PWS+20", "PWS-20"],
        # @experiment_duration: minimal experiment duration in seconds. this is used to cut out only the relevant data
        "experiment_duration": 120,
        # @data_loader: class that implements the loading of IMU data from a specific file format
        "data_loader": PhysilogDataLoader,
        # @trajectory_estimator: class that implements a trajectory estimation algorithm
        "trajectory_estimator": TuncaTrajectoryEstimator,
        # @gait_event_detector: class that implements a gait event detection algorithm
        "gait_event_detector": TuncaEventDetector,
        # @reference_loader: class that implements the loading of reference data from a specific file format.
        # OptogaitReferenceLoader and ZebrisReferenceLoader are implemented.
        # "reference_loader": ZebrisReferenceLoader,
        "reference_loader": OptogaitReferenceLoader,
        # @reference_name: name that identifies the used reference loader. This is used in plots in the evaluator
        # "reference_name": "Zebris",
        "reference_name": "OptoGait",
    }

    # instantiate the pipeline
    pipeline = Pipeline(pipeline_config)

    # the pipeline can be executed for an arbitrary subset of the dataset
    # each trial is identified by a tuple (subject_id, run_id)
    # the subject_id and run_id identify the subject and run at that position in subject and run list
    all_subjects_all_trials = [
        (x, y)
        for x in range(0, len(pipeline_config["subjects"]))
        for y in range(0, len(pipeline_config["runs"]))
    ]

    pipeline.execute(all_subjects_all_trials)
