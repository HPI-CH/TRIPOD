import numpy as np
from scipy.signal import find_peaks, peak_prominences


def gyro_threshold_stance(
    imu, stance_magnitude_threshold=0.5, stance_count_threshold=8
):
    """
    As presented by Tunca et al. (https://doi.org/10.3390/s17040825).
    Detect stance phases according to the given gyroscope magnitude, and minimum sample length threshold value.

    Args:
        imu (IMU): IMU object with data
        stance_magnitude_threshold (float): magnitude threshold
        stance_count_threshold (int): grace period in samples

    Returns:
        np.array: Boolean vector indicating stance phases
    """

    gyro = imu.gyro()

    gyro_mag = np.linalg.norm(gyro, axis=1)

    stance = gyro_mag < stance_magnitude_threshold

    stance_count = 0

    for i in range(0, len(stance)):
        if stance[i]:
            if stance_count < stance_count_threshold:
                stance[i] = False
            stance_count += 1
        else:
            if stance_count >= stance_count_threshold:
                for j in range(1, stance_count_threshold + 1):
                    stance[i - j] = False
            stance_count = 0

    return stance


def zero_crossing(x_1, x_2, y_1, y_2):
    """
    Helper function: linear interpolation for zero crossing between points (x_1, y_1) and (x_2, y_2).

    Args:
        x_1 (float): x coordinate point 1
        x_2 (float): x coordinate point 2
        y_1 (float): y coordinate point 1
        y_2 (float): y coordinate point 2

    Returns:
        float: x value for zero-crossing point
    """

    return -((y_1 * x_2 - y_2 * x_1) / (x_2 - x_1)) / ((y_2 - y_1) / (x_2 - x_1))


def hundza_gait_events(imu):
    """
    Gait event detection by Hundza et al. (https://doi.org/10.1109/TNSRE.2013.2282080)
    An example of another gait event detection algorithm.

    Args:
        imu (IMU): IMU object with data

    Returns:
        tuple[list[float], ...]: timesteps of different gait events TOFS, IOFS, TO (see Hundza paper) and stance vector
    """

    gyro = imu.gyro()
    t = imu.time()

    # select axis with highest std.dev.
    G = np.transpose(gyro)[np.argmax(gyro.std(axis=0))]

    # since left and right foot sensors are mirrored, the signal needs to be inverted.
    if len(G[G > 0]) > len(G[G <= 0]):
        G = -G

    # find local minima and maxima
    maxima, _ = find_peaks(G, prominence=1)
    minima, _ = find_peaks(-G, prominence=1)

    # filter maxima for maxima above a threshold
    threshold = 3
    positive_maxima = np.intersect1d(maxima, np.argwhere(G > threshold))

    TOFS = np.array([])  # Termination of forward swing
    IOFS = np.array([])  # Initiation of forwars swing
    TO = np.array([])  # Toe off

    stance = np.ones_like(t, dtype=bool)

    # indices of all zero or negative samples
    negative_zero_idx = np.asarray((G <= 0).nonzero())

    for maximum_id in positive_maxima:
        # find first zero crossing AFTER the maximum
        potential_tofs = negative_zero_idx[negative_zero_idx > maximum_id][0]
        if G[potential_tofs] == 0:
            TOFS = np.append(TOFS, t[potential_tofs])
            stance_begin = potential_tofs
        else:
            # if there is no sample at the zero crossing, interpolate linearly to find exact zero crossing time
            tofs = zero_crossing(
                t[potential_tofs - 1],
                t[potential_tofs],
                G[potential_tofs - 1],
                G[potential_tofs],
            )
            TOFS = np.append(TOFS, tofs)
            if tofs - t[potential_tofs - 1] < t[potential_tofs] - tofs:
                stance_begin = potential_tofs - 1
            else:
                stance_begin = potential_tofs

        # find first zero crossing BEFORE the maximum
        potential_iofs = negative_zero_idx[negative_zero_idx < maximum_id][-1]
        if G[potential_iofs] == 0:
            IOFS = np.append(IOFS, t[potential_iofs])
        else:
            # if there is no sample at the zero crossing, interpolate linearly to find exact zero crossing time
            iofs = zero_crossing(
                t[potential_iofs],
                t[potential_iofs + 1],
                G[potential_iofs],
                G[potential_iofs + 1],
            )
            IOFS = np.append(IOFS, iofs)

        # find the first minimum BEFORE the initiation of forward swing
        to = minima[minima < np.argwhere(t < iofs)[-1]][-1]
        TO = np.append(TO, t[to])

        stance[to:stance_begin] = False

    return TOFS, IOFS, TO, stance


def tunca_gait_events(imu, stance_magnitude_threshold, stance_count_threshold):
    """
    Gait event detection by Tunca et al. (https://doi.org/10.3390/s17040825)

    Args:
        imu (IMU): IMU object with data
        gyro_threshold (float): gyroscope threshold for identifying stance phases
        stance_count_threshold (int): stance count threshold for identifying stance phases

    Returns:
        tuple[list[float], ...]: index of IC_samples, FO_samples, timestamps of IC_times, FO_times and stance indicator
    """

    stance = gyro_threshold_stance(
        imu,
        stance_magnitude_threshold=stance_magnitude_threshold,
        stance_count_threshold=stance_count_threshold,
    )
    step_begins = np.where(np.logical_and(np.logical_not(stance[:-1]), stance[1:]))[0]

    gyro = imu.gyro()
    t = imu.time()

    # select axis with highest std.dev.
    G = np.transpose(gyro)[np.argmax(gyro.std(axis=0))]

    # since left and right foot sensors are mirrored, the signal needs to be inverted.
    if len(G[G > 0]) > len(G[G <= 0]):
        G = -G

    # calculate tilt
    dt = np.diff(t)
    tilt = np.cumsum(G[1:] * dt)

    # recalculate tilt without bias
    tilt_diff_bias = tilt[-1] / (t[-1] - t[0])
    tilt_diff = G - tilt_diff_bias
    tilt = np.cumsum(tilt_diff[1:] * dt)

    # get start and end of search regions
    fo_search_region_ends, _ = find_peaks(-tilt, prominence=0.1)
    ic_search_region_starts, _ = find_peaks(tilt, prominence=0.1)

    # get FO and IC candidates
    fo_ic_candidates, _ = find_peaks(-tilt_diff, prominence=0.1)
    prominences = peak_prominences(-tilt_diff, fo_ic_candidates)[0]

    FO_samples = np.array([], dtype=int)
    IC_samples = np.array([], dtype=int)
    FO_prominences = np.array([])
    IC_prominences = np.array([])

    for step_begin, step_end in zip(step_begins[:-1], step_begins[1:]):

        # identify search region boundaries within one step
        fo_search_region_end = fo_search_region_ends[
            np.logical_and(
                fo_search_region_ends > step_begin, fo_search_region_ends <= step_end
            )
        ]
        ic_search_region_start = ic_search_region_starts[
            np.logical_and(
                ic_search_region_starts > step_begin,
                ic_search_region_starts <= step_end,
            )
        ]

        # there shoud be exactly one FO and one IC search region boundary and the search regions should not overlap,
        # otherwise events cannot be detected properly
        if (
            len(fo_search_region_end) == 1
            and len(ic_search_region_start) == 1
            and ic_search_region_start >= fo_search_region_end
        ):
            fo_search_region_end = fo_search_region_end[0]
            ic_search_region_start = ic_search_region_start[0]
        else:
            # overlaping or unconclusive search regions
            # if more than one boundary detected: ignore all events for this step
            continue

        fo_cand, _ = find_peaks(
            -tilt_diff[step_begin:fo_search_region_end], prominence=0.1
        )
        ic_cand, _ = find_peaks(
            -tilt_diff[ic_search_region_start:step_end], prominence=0.1
        )
        fo_prominence_cand = peak_prominences(
            -tilt_diff[step_begin:fo_search_region_end], fo_cand
        )[0]
        ic_prominence_cand = peak_prominences(
            -tilt_diff[ic_search_region_start:step_end], ic_cand
        )[0]

        # transform step-local indices of the candidates to full signal indices
        fo_cand = np.asarray([x + step_begin for x in fo_cand])
        ic_cand = np.asarray([x + ic_search_region_start for x in ic_cand])

        # identify the first event before and after the search region boundary
        if np.any(
            np.logical_and(fo_cand < fo_search_region_end, fo_cand >= step_begin)
        ) and np.any(
            np.logical_and(ic_cand > ic_search_region_start, ic_cand < step_end)
        ):
            stride_fo_cand = fo_cand[np.logical_and(fo_cand < fo_search_region_end, fo_cand >= step_begin)]
            stride_ic_cand = ic_cand[np.logical_and(ic_cand > ic_search_region_start, ic_cand < step_end)]
            FO_samples = np.append(
                FO_samples,
                fo_cand[np.argmin(tilt_diff[stride_fo_cand])],
            )
            IC_samples = np.append(
                IC_samples,
                ic_cand[0],
            )

            stride_fo_prominence_cand = fo_prominence_cand[np.logical_and(fo_cand < fo_search_region_end, fo_cand >= step_begin)]
            stride_ic_prominence_cand = ic_prominence_cand[np.logical_and(ic_cand > ic_search_region_start, ic_cand < step_end)]
            FO_prominences = np.append(
                FO_prominences,
                stride_fo_prominence_cand[np.argmin(tilt_diff[stride_fo_cand])],
            )
            IC_prominences = np.append(
                IC_prominences,
                ic_prominence_cand[0],
            )
        else:
            # No ic or fo found in search region
            continue

    IC_times = [t[sample] if not np.isnan(sample) else np.nan for sample in IC_samples]
    FO_times = [t[sample] if not np.isnan(sample) else np.nan for sample in FO_samples]

    return IC_samples, FO_samples, IC_times, FO_times, stance
