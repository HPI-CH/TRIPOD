""" This module contains various implementations of trajectory estimators and necessary helper functions. """

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as rot
from event_detection.imu_event_detection import gyro_threshold_stance


def gyro_to_euler(orientation, gyro):
    """
    Applies a 3D gyroscope measurement in the local frame (deg/s) to a global 3D orientation vector (rad)
    and calculates the 3D rotation vector (rad/s) in the global coordinate frame.

    Args:
        orientation (np.array): initial orientation vector
        gyro (np.array): gyroscope signal vector

    Returns:
        np.array: rotation vector
    """

    orientation = orientation.flatten()

    # transformation matrices
    roll = np.matrix(
        [
            [1, 0, 0],
            [0, np.cos(orientation[0]), -np.sin(orientation[0])],
            [0, np.sin(orientation[0]), np.cos(orientation[0])],
        ]
    )

    pitch = np.matrix(
        [
            [np.cos(orientation[1]), 0, np.sin(orientation[1])],
            [0, 1, 0],
            [-np.sin(orientation[1]), 0, np.cos(orientation[1])],
        ]
    )

    yaw = np.matrix(
        [
            [np.cos(orientation[2]), -np.sin(orientation[2]), 0],
            [np.sin(orientation[2]), np.cos(orientation[2]), 0],
            [0, 0, 1],
        ]
    )

    return roll @ pitch @ yaw @ gyro * np.pi / 180


def acc_to_euler(acc):
    """
    Calculate the 3D orientation vector (rad) from 3D acceleration measurement,
    assuming that there is no translational movement.
    Note: the z-coordinate is always zero since horizontal rotation cannot be tracked by an accelerometer.

    Args:
        acc (np.array): acceleration measurement

    Returns:
        np.array: oriantation vector
    """

    orientation_est_acc = np.zeros((3, 1))
    orientation_est_acc[0] = np.arctan(
        acc[1] / np.sqrt(np.square(acc[0]) + np.square(acc[2]))
    )
    orientation_est_acc[1] = np.arctan(
        acc[0] / np.sqrt(np.square(acc[1]) + np.square(acc[2]))
    )

    return orientation_est_acc


def complementary_filter(time, acc, gyro, a=0.1):
    """
    Complementary filter to fusion accelerometer and gyroscope measurements.
    Calculate the deduced orientation time series from accelerometer and gyroscope
    Note: a quantifies the "trust" for the accelerometer or the "misstrust" for the gyroscope.

    Args:
        time (np.array): timestamps
        acc (np.array): 3D acceleration measurements
        gyro (np.array): 3D gyroscope measurements
        a (float): complementary factor

    Returns:
        np.array: Orientation vector timeseries
    """

    orientation_est = np.zeros((len(time) + 1, 3, 1))

    for i in range(0, len(time)):
        d_rot = gyro_to_euler(orientation_est[i], gyro[i].reshape(3, 1))

        if i == 0:
            dt = time[i]
        else:
            dt = time[i] - time[i - 1]

        orientation_est[i + 1] = (1 - a) * (
            orientation_est[i] + d_rot * dt
        ) + a * acc_to_euler(acc[i])

    return orientation_est * (180 / np.pi)


def kalman_filter(time, acc, gyro):
    """
    Kalman filter implementation inspired by http://philsal.co.uk/projects/imu-attitude-estimation.

    Args:
        time (np.array): timestamps
        acc (np.array):  3D acceleration measurements
        gyro (np.array): 3D gyroscope measurements

    Returns:
        list[list[float]]: 3D position timeseries
    """

    # time series of state vectors
    # [[roll], [pitch], [yaw], [roll_bias], [pitch_bias], [yaw_bias]]
    state = np.zeros((len(time) + 1, 6, 1))

    # measurement Matrix
    # select the angles and drops the bias
    C = np.matrix([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])

    # Covariance Matrix
    P = np.eye(6)

    # noise covariance
    Q = np.eye(6)

    # measurement variance
    R = np.eye(3)

    for i in range(0, len(time)):
        if i == 0:
            dt = time[i]
        else:
            dt = time[i] - time[i - 1]

        # dynamic matrix
        # update angles with gyro bias
        A = np.matrix(
            [
                [1, 0, 0, -dt, 0, 0],
                [0, 1, 0, 0, -dt, 0],
                [0, 0, 1, 0, 0, -dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # steering matrix
        # update angles with gyro input
        B = np.matrix(
            [
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt],
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        )

        d_rot = gyro_to_euler(state[i][0:3], gyro[i].reshape(3, 1))

        # predict
        state[i + 1] = A @ state[i] + B @ d_rot
        P = A @ P @ np.transpose(A) + Q

        # update
        w = acc_to_euler(acc[i]) - C @ state[i + 1]
        S = C @ P @ np.transpose(C) + R
        K = P @ np.transpose(C) @ np.linalg.inv(S)
        state[i + 1] = state[i + 1] + K @ w
        P = (np.eye(6) - K @ C) @ P

    return [s[0:3] * (180 / np.pi) for s in state]


def error_state_kalman_filter(  # noqa: C901
    imu,
    imu_ic,
    zero_z=True,
    zero_xz=False,
    zero_xyz=False,
    stance_magnitude_threshold=0.5,
    stance_count_threshold=8,
    method="euler",
):
    """
    Error-state Kalman Filter inspired by https://doi.org/10.3390/s17040825.

    Args:
        imu (IMU): IMU object with measurement data
        imu_ic (float): timestamp of initial contact
        zero_z (bool): flag if height should be reset each stride
        zero_xz (bool): flag if height and sideward movement should be reset each stride (if true, zero_z is ignored)
        zero_xyz (bool) flag if all position dimensions should be reset each stride (if true, zero_z and zero_xz are ignored)
        stance_magnitude_threshold (float): gyroscope magnitude threshold for stance event_detection
        stance_count_threshold (int): grace period in samples before and after stance detection
        method (str): "euler" or "midpoint", select used integration technique

    Returns:
        DataFrame: DataFrame with trajectory data for each timestep (time, position, velocity, rotation, stance)
    """

    stance = gyro_threshold_stance(
        imu, stance_magnitude_threshold, stance_count_threshold
    )

    time = imu.time()
    acc = imu.accel()
    gyro = imu.gyro()

    stance_begins = np.where(np.logical_and(np.logical_not(stance[:-1]), stance[1:]))[0]

    ic_sample = np.flatnonzero(time >= imu_ic)[0]
    # start at the fourth step (gait should be stable by then)
    start_sample = stance_begins[stance_begins > ic_sample][4]

    # enable RTS smoother
    rts_enabled = True

    # Set known error distributions for sensors
    acc_sigma = 0.05
    gyro_sigma = 0.02
    v_meas_sigma = 1e-4  # velocity uncertainty
    z_meas_sigma = 1e-4  # displacement uncertainty

    # State
    C = np.zeros((len(time), 3, 3))  # orientation (3x3 rotation matrix)
    v = np.zeros((len(time), 3))  # velocity (3D vector)
    s = np.zeros((len(time), 3))  # displacement (3D vector)

    # F error-state update matrix
    F = np.zeros((len(time), 9, 9))  # 3x3x(3x3 matrix)

    # P error-state covariance matrix
    P = np.zeros((len(time), 9, 9))  # 3x3x(3x3 matrix)

    # initialize error-state
    dx = np.zeros((len(time), 9, 1))
    dx_rts = np.zeros((len(time), 9, 1))

    # initial covarriance of orientation
    V = np.array([[1e-3, 0, 0], [0, 1e-3, 0], [0, 0, 1e-10]])
    # initial covariance of position
    W = 1e-5 * np.eye(3)

    orientation_init = False

    for t in range(start_sample, len(time)):
        # dt = time[t] - time[t-1]
        dt = 0.0078125

        if not orientation_init:
            if stance[t - 1]:
                # initialize orientation (using only accelerometer)
                up = np.array([0, 1, 0])  # up-pointing vector as reference

                x_axis = np.cross(up, acc[t - 1])
                x_axis = x_axis / np.linalg.norm(x_axis)

                y_axis = np.cross(acc[t - 1], x_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)

                init_acc = acc[t - 1] / np.linalg.norm(acc[t - 1])

                C[t - 1][0] = x_axis
                C[t - 1][1] = y_axis
                C[t - 1][2] = init_acc

                P[t - 1][0:3, 0:3] = V
                P[t - 1][3:6, 3:6] = W

                orientation_init = True
            else:
                continue

        if np.linalg.norm(gyro[t]) == 0:
            gyro[t] = 1e-10 * np.ones(3, 1)

        # Orientation update
        # see: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle

        # define the cross product matrix B for the rotation gyro rotation axis
        B = np.array(
            [
                [0, -gyro[t][2], gyro[t][1]],
                [gyro[t][2], 0, -gyro[t][0]],
                [-gyro[t][1], gyro[t][0], 0],
            ]
        )

        # get the actual turning angle around the turning axis
        angle = np.linalg.norm(gyro[t] * dt)
        B = B * dt  # turn angular velocity into angles
        B = B / angle  # normalize

        # calculate rotation matrix with "Rodrigues' rotation formula"
        R = np.eye(3) + np.sin(angle) * B + (1 - np.cos(angle)) * B @ B
        # apply rotation to previous orientation
        C[t] = C[t - 1] @ R

        # calculate total acceleration force in the local frame
        fg = (C[t] @ acc[t].reshape(3, 1)).reshape(3)
        # subtract gravitation in order to get pure acceleration
        ag = fg - np.array([0, 0, 9.807])

        if method == "euler":  # really euler-cromer
            # calculate velocity
            v[t] = v[t - 1] + ag * dt
            # calculate displacement
            s[t] = s[t - 1] + v[t] * dt
        elif method == "midpoint":
            # calculate velocity
            v[t] = v[t - 1] + ag * dt
            # calculate displacement
            s[t] = s[t - 1] + 0.5 * (v[t] + v[t - 1]) * dt

        # Error-state propagation
        # define the cross product matrix S for the accelerometer axis
        S = np.array(
            [
                [0, -fg[2], fg[1]],
                [fg[2], 0, -fg[0]],
                [-fg[1], fg[0], 0],
            ]
        )
        # everything evolves linearily
        F[t] = np.eye(9)
        # ... except the rotation influences the acceleration
        F[t][3:6, 0:3] = S * dt
        # ... except the acceleration influences the position with factor dt
        F[t][6:9, 3:6] = np.eye(3) * dt

        # gyro system uncertainty
        T = np.transpose(C[t]) * dt ** 2 @ (np.eye(3) * gyro_sigma ** 2) @ C[t]
        # accelerometer uncertainty
        U = np.transpose(C[t]) * dt ** 2 @ (np.eye(3) * acc_sigma ** 2) @ C[t]

        Q = np.zeros((9, 9))
        Q[0:3, 0:3] = T
        Q[3:6, 3:6] = U

        P[t] = F[t] @ P[t - 1] @ np.transpose(F[t]) + Q

        # error-state correction during stance phase
        if stance[t]:
            stance_v = np.zeros(3)

            if zero_xyz:
                # correct velocity
                # correct xyz-position coordinate to 0
                H = np.array(
                    [
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                )
                # R provides a noise estimation of the measurement
                R = np.array(
                    [
                        [v_meas_sigma ** 2, 0, 0, 0, 0, 0],
                        [0, v_meas_sigma ** 2, 0, 0, 0, 0],
                        [0, 0, v_meas_sigma ** 2, 0, 0, 0],
                        [0, 0, 0, z_meas_sigma ** 2, 0, 0],
                        [0, 0, 0, 0, z_meas_sigma ** 2, 0],
                        [0, 0, 0, 0, 0, z_meas_sigma ** 2],
                    ]
                )

                y = np.append(-v[t] + stance_v, -s[t]).reshape(6, 1)

            elif zero_z:
                # correct velocity and z-position coordinate to 0
                # H identifies the velocity components and the z-position coordinate in the state vector
                H = np.array(
                    [
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                )
                # R provides a noise estimation of the measurement
                R = np.array(
                    [
                        [v_meas_sigma ** 2, 0, 0, 0],
                        [0, v_meas_sigma ** 2, 0, 0],
                        [0, 0, v_meas_sigma ** 2, 0],
                        [0, 0, 0, z_meas_sigma ** 2],
                    ]
                )
                # "Fake" velocity and z-position measurement data that is later on added to the predicted state
                # and thus leads to zero velocity and z-position
                y = np.append(-v[t] + stance_v, -s[t][2]).reshape(4, 1)

            elif zero_xz:
                # correct velocity
                # correct x and z position to 0. This is especially relevant for treadmill walking
                H = np.array(
                    [
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    ]
                )
                # R provides a noise estimation of the measurement
                R = np.array(
                    [
                        [v_meas_sigma ** 2, 0, 0, 0, 0],
                        [0, v_meas_sigma ** 2, 0, 0, 0],
                        [0, 0, v_meas_sigma ** 2, 0, 0],
                        [0, 0, 0, z_meas_sigma ** 2, 0],
                        [0, 0, 0, 0, z_meas_sigma ** 2],
                    ]
                )

                y = np.append(-v[t] + stance_v, -s[t][[0, 2]]).reshape(5, 1)

            else:
                # only correct velocity to 0
                # H identifies the velocity components in the state vector
                H = np.array(
                    [
                        [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    ]
                )
                # R provides a noise estimation of the measurement
                R = np.array(
                    [
                        [v_meas_sigma ** 2, 0, 0],
                        [0, v_meas_sigma ** 2, 0],
                        [0, 0, v_meas_sigma ** 2],
                    ]
                )
                # "Fake" velocity measurement data that is later on added to the predicted state
                # and thus leads to zero velocity
                y = (-v[t] + stance_v).reshape(3, 1)

            # Calculate Kalman gain from H, P and R
            K = P[t] @ np.transpose(H) @ np.linalg.inv(H @ P[t] @ np.transpose(H) + R)

            # calculate error state by applying the Kalman gain to the "fake" measurement data
            dx[t] = K @ y

            # apply error-state to state vector (velocity and displacement)
            v[t] = v[t] + dx[t].reshape(9)[3:6]
            s[t] = s[t] + dx[t].reshape(9)[6:9]

            # rotation error-state
            phi = dx[t].reshape(9)[0]
            theta = dx[t].reshape(9)[1]
            psi = dx[t].reshape(9)[2]

            # calculate rotation matrix
            rot_corr = np.array(
                [
                    [
                        np.cos(theta) * np.cos(psi),
                        np.cos(theta) * np.sin(psi),
                        -np.sin(theta),
                    ],
                    [
                        np.sin(phi) * np.sin(theta) * np.cos(psi)
                        - np.cos(phi) * np.sin(psi),
                        np.sin(phi) * np.sin(theta) * np.sin(psi)
                        + np.cos(phi) * np.cos(psi),
                        np.sin(phi) * np.cos(theta),
                    ],
                    [
                        np.cos(phi) * np.sin(theta) * np.cos(psi)
                        + np.sin(phi) * np.sin(psi),
                        np.cos(phi) * np.sin(theta) * np.sin(psi)
                        - np.sin(phi) * np.cos(psi),
                        np.cos(phi) * np.cos(theta),
                    ],
                ]
            )

            # apply rotation matrix
            C[t] = rot_corr @ C[t]

            # RTS smoothing
            if rts_enabled:
                dx_rts[t] = dx[t]
                k = t - 1
                while k > 0 and not stance[k]:
                    G = P[k] @ np.transpose(F[k + 1]) @ np.linalg.inv(P[k + 1])
                    dx_rts[k] = G @ dx_rts[k + 1]

                    # Error-state transfer
                    v[k] += dx_rts[k].reshape(9)[3:6]
                    s[k] += dx_rts[k].reshape(9)[6:9]

                    dphix = dx_rts[k][0]
                    dphiy = dx_rts[k][1]
                    dphiz = dx_rts[k][2]
                    phi = np.array(
                        [
                            [1, dphiz, -dphiy],
                            [-dphiz, 1, dphix],
                            [dphiy, -dphix, 1],
                        ]
                    )
                    C[k] = phi.astype(float) @ C[k]

                    k -= 1

            # update error state covariance matrix
            P[t] = (np.eye(9) - K @ H) @ P[t]

    quat_rotation = np.transpose([rot.from_dcm(c).as_quat() for c in C])

    return pd.DataFrame(
        data={
            "time": time,
            "position_x": np.transpose(s)[0],
            "position_y": np.transpose(s)[1],
            "position_z": np.transpose(s)[2],
            "velocity_x": np.transpose(v)[0],
            "velocity_y": np.transpose(v)[1],
            "velocity_z": np.transpose(v)[2],
            "rotation_w": quat_rotation[0],
            "rotation_x": quat_rotation[1],
            "rotation_y": quat_rotation[2],
            "rotation_z": quat_rotation[3],
            "stance": stance,
        }
    )
