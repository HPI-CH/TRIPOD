""" This module contains various plotting functions.
    Not all of them are actually used in the pipeline but could be potentially useful.
    Some plotting functions can be stacked, resulting in complex plots.
    Therefore, show() needs to be called manually to display the plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  # NOQA


def plot_1d(y, x=None):
    """Plot a 1D array. Generate regular spacing if x is None.

    Args:
        y (list[float]): y values
        x (list[float]): optional - value spacing

    Returns:
        None
    """

    plt.figure()
    if x is not None:
        plt.plot(x, y)
    else:
        plt.plot(np.arange(0, len(y)), y)


def plot_accel_gyro(imu):
    """Plots acceleration and gyroscope raw values.

    Args:
        imu (IMU): IMU object with data to plot

    Returns:
        None
    """

    _, axarr = plt.subplots(2, sharex=True)

    axarr[0].plot(imu.time(), imu.data["Accel X"], label="X")
    axarr[0].plot(imu.time(), imu.data["Accel Y"], label="Y")
    axarr[0].plot(imu.time(), imu.data["Accel Z"], label="Z")
    axarr[0].set_title("Acceleration")
    axarr[0].legend()
    axarr[0].grid()

    axarr[1].plot(imu.time(), imu.data["Gyro X"], label="X")
    axarr[1].plot(imu.time(), imu.data["Gyro Y"], label="Y")
    axarr[1].plot(imu.time(), imu.data["Gyro Z"], label="Z")
    axarr[1].set_title("Gyroscope")
    axarr[1].legend()
    axarr[1].grid()


def plot_imu_accel_mag(imu):
    """Plots acceleration magnitude.

    Args:
        imu (IMU): IMU object with data to plot

    Returns:
        None
    """

    plt.figure()

    plt.plot(imu.time(), np.linalg.norm(imu.accel(), axis=1))
    plt.title("Acceleration Magnitude")
    plt.grid()


def plot_stance_overlay(time, stance):
    """
    Plot transparent overlay during stance phases.
    This makes it easy to visualize stance phases in any signal plot and adapt the
    threshold that is used to identify these phases.

    Args:
        stance (np.array): bool array, indicating stance for each timestep

    Returns:
        None
    """

    stance_begin_end = np.where(stance[:-1] != stance[1:])[0]
    if stance[0]:
        stance_begin_end = np.append([0], stance_begin_end)
    if np.mod(len(stance_begin_end), 2):
        stance_begin_end = np.append(stance_begin_end, len(stance) - 1)
    for i in range(0, len(stance_begin_end), 2):
        plt.axvspan(
            time[stance_begin_end[i]],
            time[stance_begin_end[i + 1]],
            facecolor="b",
            alpha=0.5,
        )


def plot_gyro_magnitude(imu, stance=None, threshold=None):
    """Plot gyroscope magnitude

    Args:
        imu (IMU): IMU object with data
        stance (np.array): optional stance indicators. If set, an overlay is plotted
        threshold (float): optional - If set, a horizontal line at the specified height is drawn

    Returns:
        None
    """

    fig = plt.figure()
    gyro_mag = np.linalg.norm(imu.gyro(), axis=1)
    plt.plot(imu.time(), gyro_mag)
    if stance is not None:
        plot_stance_overlay(imu.time(), stance)

    if threshold is not None:
        plt.hlines(threshold, imu.time()[0], imu.time()[-1])

    plt.grid()
    plt.title("Stance detection")
    plt.ylabel("Gyroscope magnitude [rad/s]")
    plt.xlabel("Time [s]")

    return fig


def set_3d_axes_equal(ax, data):
    """Sets all axes to an equal scale.
    Note: ax.set_aspect('equal') is not available for 3D.

    Args:
        ax (matplotlib.axes): Axes of the plot
        data (np.array): 3D data that is plotted along the axes

    Returns:
        None
    """

    data_min = np.amin(data, axis=1)
    data_max = np.amax(data, axis=1)

    axes_range = data_max - data_min
    bounding_box_size = np.max(axes_range)

    for x in range(0, 3):
        data_min[x] -= (bounding_box_size - axes_range[x]) / 2
        data_max[x] += (bounding_box_size - axes_range[x]) / 2

    ax.set_xlim3d([data_min[0], data_max[0]])
    ax.set_ylim3d([data_min[1], data_max[1]])
    ax.set_zlim3d([data_min[2], data_max[2]])


def set_3d_xy_axes_equal(ax, data):
    """Sets horizontal axes to an equal scale.
    This is relevant for displaying foot trajectories.
    Since the height of a step is very small compared to the length,
    horizontal and vertical axes should be scaled differently in order to
    see actual change along the vertical axis.

    Args:
        ax (matplotlib.axes): Axes of the plot
        data (np.array): 3D data that is plotted along the axes

    Returns:
        None
    """

    data_min = np.amin(data, axis=1)
    data_max = np.amax(data, axis=1)

    axes_range = data_max - data_min
    bounding_box_size = np.max(axes_range)

    for x in range(0, 2):
        data_min[x] -= (bounding_box_size - axes_range[x]) / 2
        data_max[x] += (bounding_box_size - axes_range[x]) / 2

    ax.set_xlim3d([data_min[0], data_max[0]])
    ax.set_ylim3d([data_min[1], data_max[1]])
    ax.set_zlim3d([data_min[2], data_max[2]])


def plot_3d_view(position):
    """Plots 3D trajectory.

    Args:
        position (np.array): 3D coordinates for each timestep

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    step_data = [
        [x[0] for x in position],
        [x[1] for x in position],
        [x[2] for x in position],
    ]

    ax.scatter(step_data[0], step_data[1], step_data[2])
    ax.plot(step_data[0], step_data[1], step_data[2])
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Position y [m]")
    ax.set_zlabel("Position z [m]")

    set_3d_xy_axes_equal(ax, step_data)


def plot_3d_view_with_stance(position, stance):
    """Plots trajectory in 3D and colors stance phases differently.

    Args:
        position (np.array): 3D position for each timestep
        stance (np.array): bool array indicating stance for each timestep

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    step_data = [
        [x[0] for x in position[np.logical_not(stance)]],
        [x[1] for x in position[np.logical_not(stance)]],
        [x[2] for x in position[np.logical_not(stance)]],
    ]
    stance_data = [
        [x[0] for x in position[stance]],
        [x[1] for x in position[stance]],
        [x[2] for x in position[stance]],
    ]

    ax.scatter(step_data[0], step_data[1], step_data[2])
    ax.scatter(stance_data[0], stance_data[1], stance_data[2])
    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Position y [m]")
    ax.set_zlabel("Position z [m]")

    set_3d_axes_equal(
        ax,
        [[x[0] for x in position], [x[1] for x in position], [x[2] for x in position]],
    )


def plot_multi_3d_view(positions):
    """Plots multiple trajectories in one 3D graph.

    Args:
        positions (list[np.array]): multiple arrays of 3D coordinates for each timestep

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for position in positions:
        ax.plot(
            [x[0] for x in position], [x[1] for x in position], [x[2] for x in position]
        )

    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Position y [m]")
    ax.set_zlabel("Position z [m]")

    set_3d_xy_axes_equal(ax, np.concatenate(positions, axis=0).T)


def plot_3d_view_speed(position, speed):
    """Plots 3D trajectory colored by speed.

    Args:
        position (np.array): 3D position for each timestep
        speed (np.array): speed for each timestep

    Returns:
        None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("gnuplot")

    speed_norm = np.linalg.norm(speed, axis=1)
    speed_norm = (speed_norm - np.min(speed_norm)) / (
        np.max(speed_norm) - np.min(speed_norm)
    )

    for i in range(0, len(position) - 1):
        ax.plot(
            [position[i][0], position[i + 1][0]],
            [position[i][1], position[i + 1][1]],
            [position[i][2], position[i + 1][2]],
            color=cmap(speed_norm[i]),
        )

    ax.set_xlabel("Position x [m]")
    ax.set_ylabel("Position y [m]")
    ax.set_zlabel("Position z [m]")

    set_3d_xy_axes_equal(
        ax,
        [[x[0] for x in position], [x[1] for x in position], [x[2] for x in position]],
    )


def plot_2d_lateral_view(position, stance=None, as_subplot=False):
    """
    Plots trajectory in 2D (lateral view, height and distance from origin).
    Optionally mark stance phase.
    Optionally add this as subplot. Used to overlay multiple steps.

    Args:
        position (np.array): 3D position for each timestep
        stance (np.array): optional bool array, indicating stance phase
        as_subplot (bool): used if called from plot_steps()

    Returns:
        None
    """

    if not as_subplot:
        plt.figure()
    if stance is None:
        stance = np.array([False] * len(position))
    plt.plot(
        [np.linalg.norm(x[0:2]) for x in position[np.logical_not(stance)]],
        [x[2] for x in position[np.logical_not(stance)]],
    )
    plt.scatter(
        [np.linalg.norm(x[0:2]) for x in position[stance]],
        [x[2] for x in position[stance]],
    )
    plt.grid("on")
    plt.title("2D trajectory")


def plot_steps(steps):
    """
    Plots a 2D view of overlayed steps.
    This can be used to compare the shape of trajectories of different steps.

    Args:
        steps (list[np.array]): multiple arrays of 3D coordinates for each timestep

    Returns:
        None
    """

    plt.figure()
    for step in steps:
        plot_2d_lateral_view(step, as_subplot=True)
    plt.title("2D trajectory superposition")
    plt.ylabel("Vertical distance [m]")
    plt.xlabel("Horizontal distance [m]")


def show():
    """ Shows the plot. Proxy for plt.show() so matplotlib.pyplot doesnt have to be imported. """

    plt.show()
