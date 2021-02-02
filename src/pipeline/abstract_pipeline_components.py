"""This module implements the abstract classes/interfaces of all variable pipeline components."""


class AbstractDataLoader:
    """The AbstractDataLoader defines the interface for any kind of IMU data loader."""

    def __init__(self, raw_base_path, dataset, subject, run):
        """
        Initialization of an AbstractDataLoader

        Args:
            raw_base_path (str): Base folder for every dataset
            dataset (str): Folder containing the dataset
            subject (str): Subject identifier
            run (str): Run identifier
        """
        self.data = {}
        self.load(raw_base_path, dataset, subject, run)

    def load(self, raw_base_path, dataset, subject, run):
        """
        This method is called at instantiation and needs to be implemented by the derived class.
        It is expected to find the actual data files based on the given parameters
        and store them as a dictionary of IMU objects into self.data

        Args:
            raw_base_path (str): Base folder for every dataset
            dataset (str): Folder containing the dataset
            subject (str): Subject identifier
            run (str): Run identifier

        Returns:
            None

        """
        pass

    def get_data(self):
        """
        Get the loaded data.

        Returns:
            dict[str, IMU]: IMU data from all sensors
        """
        return self.data


class AbstractTrajectoryEstimator:
    """ The AbstractTrajectoryEstimator defines the interface for each trajectory estimation algorithm."""

    def __init__(self, imus):
        """
        Initialization of an AbstractTrajectoryEstimator

        Args:
            imus dict[str, IMU]: Dictionary of IMU objects for each sensor location
        """
        self.imus = imus

    def estimate(self, interim_base_path, dataset, subject_num, run_num):
        """
        This method is expected to be implemented by each trajectory estimation algorithm.

        Args:
            interim_base_path (str): Base folder where interim data can be stored
            dataset (str): Folder containing the dataset
            subject_num (int): Subject index
            run_num (int): Run index

        Returns:
            dict[str, DataFrame]: DataFrames containing trajectory information for each foot

        """
        pass


class AbstractEventDetector:
    """
    The AbstractEventDetector defines the interface for any kind of event detection algorithm.
    """

    def __init__(self, imus):
        """
        Initialization of an AbstractEventDetector.

        Args:
            imus (dict[str, IMU]): Dictionary of IMU objects for each sensor location.
        """
        self.imus = imus

    def detect(self):
        """
        This method is expected to be implemented by each event detection algorithm.

        Returns:
            (dict[str, dict]): dictionaries containing gait event information for each foot.

        """
        pass


class AbstractReferenceLoader:
    """
    The AbstractReferenceLoader defines the interface for any kind of reference data loader.
    """

    def __init__(
        self, name, raw_base_path, interim_base_path, dataset, subject, run, overwrite
    ):
        """
        Initialization of an AbstractReferenceLoader.

        Args:
            mame (str): Identifier used to create caching files
            raw_base_path (str): Base folder for every dataset
            interim_base_path (str): Base folder where interim data can be stored
            dataset (str): Folder containing the dataset
            subject (str): Subject identifier
            run (str): Run identifier
        """

        self.name = name
        self.raw_base_path = raw_base_path
        self.interim_base_path = interim_base_path
        self.dataset = dataset
        self.subject = subject
        self.run = run
        self.overwrite = overwrite
        self.data = {"left": {}, "right": {}}
        self.load()

    def load(self):
        """
        This method is called at instantiation and needs to be implemented by the derived class.
        It is expected to find the actual data files based on the given parameters
        and store reference data for the left and right foot in self.data

        Returns:
            None
        """
        pass

    def get_data(self):
        """
        Get the loaded reference data.

        Returns:
            dict[str, DataFrame]: DataFrames with gait parameters for the left and right foot
        """
        return self.data
