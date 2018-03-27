"""
Code based on Srini Ananthakrishnan's implementation
https://github.com/srianant/kalman_filter_multi_object_tracking/blob/master/kalman_filter.py
"""

import numpy as np


# noinspection PyPep8Naming
class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of the system and the variance or uncertainty of
    the estimate.
    Constant velocity model, defined over a 2D space.
    """

    def __init__(self, init_location, init_estimate_error, motion_model_noise, measurement_noise, dt=1):
        # Assertions
        init_location = np.array(init_location)
        assert init_location.shape == (2,), 'Initial location should be a 2D vector: [x, y]'

        init_estimate_error = np.array(init_estimate_error)
        assert init_estimate_error.shape == (2,), \
            'Error of the initial estimate should be a 2D vector: [init_location_var, init_velocity_var]'

        motion_model_noise = np.array(motion_model_noise)
        assert motion_model_noise.shape == (2,), \
            'Motion model noise should be a 2D vector: [location_var, velocity_var]'

        measurement_noise = float(measurement_noise)
        assert isinstance(measurement_noise, float), 'Measurement noise should be a scalar value: [measure_var]'

        # Constant velocity model, initialize velocities to 0
        self.x = np.array([init_location[0], 0, init_location[1], 0])

        # Placeholder for last measurement
        self.z = None
        # Placeholder for last predicted state
        self.last_x = None

        # Constant velocity motion model
        self.dt = dt  # delta time
        self.D = np.array(
            [[1., self.dt, 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., self.dt],
             [0., 0., 0., 1.]]
        )

        # Measurement model that relates current state to measurement z
        self.M = np.array(
            [[1., 0., 0., 0.],
             [0., 0., 1., 0.]]
        )

        # Covariance matrix
        self.sigma_k = np.diag(
            [init_estimate_error[0], init_estimate_error[1], init_estimate_error[0], init_estimate_error[1]]
        )
        # Process noise matrix
        self.sigma_d = np.diag(
            [motion_model_noise[0], motion_model_noise[1], motion_model_noise[0], motion_model_noise[1]]
        )
        # Observation noise matrix
        self.sigma_m = measurement_noise * np.eye(2)

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.x = np.round(np.dot(self.D, self.x))
        # Predicted estimate covariance
        self.sigma_k = np.dot(self.D, np.dot(self.sigma_k, self.D.T)) + self.sigma_d
        self.last_x = self.x  # same last predicted result
        return self.x

    def correct(self, z, use_detection):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            z: vector of observations
            use_detection: if "true" prediction result will be updated else detection
        Return:
            predicted state vector x
        """

        if use_detection:  # update using detection
            self.z = z
        else:  # update using prediction
            self.z = np.dot(self.M, self.last_x)
        aux_matrix_C = np.dot(self.M, np.dot(self.sigma_k, self.M.T)) + self.sigma_m
        kalman_gain = np.dot(self.sigma_k, np.dot(self.M.T, np.linalg.inv(aux_matrix_C)))

        self.x = np.round(self.x + np.dot(kalman_gain, (self.z - np.dot(self.M, self.x))))
        self.sigma_k = self.sigma_k - np.dot(kalman_gain, np.dot(aux_matrix_C, kalman_gain.T))
        self.last_x = self.x
        return self.x
