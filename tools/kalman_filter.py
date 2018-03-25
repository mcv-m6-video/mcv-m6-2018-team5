"""
Code based on Srini Ananthakrishnan's implementation
https://github.com/srianant/kalman_filter_multi_object_tracking/blob/master/kalman_filter.py
"""

import numpy as np


# noinspection PyPep8Naming
class KalmanFilter(object):
    """Kalman Filter class keeps track of the estimated state of the system and the variance or uncertainty of
    the estimate. Predict and Correct methods implement the functionality.
    Reference: https://en.wikipedia.org/wiki/Kalman_filter
    """

    def __init__(self):
        self.dt = 0.005  # delta time

        self.x = np.zeros((2, 1))  # state vector
        self.z = np.array([[0], [255]])  # observations vector

        self.D = np.array([[1.0, self.dt], [0.0, 1.0]])  # dynamics model
        self.M = np.array([[1, 0], [0, 1]])  # measurement matrix

        self.sigma_k = np.diag((3.0, 3.0))  # covariance matrix
        self.sigma_d = np.eye(self.u.shape[0])  # process noise matrix
        self.sigma_m = np.eye(self.b.shape[0])  # observation noise matrix

        self.last_x = np.array([[0], [255]])  # placeholder for last predicted state

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
        Args:
            None
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
            self.z = self.last_x
        aux_matrix_C = np.dot(self.M, np.dot(self.sigma_k, self.M.T)) + self.sigma_m
        kalman_gain = np.dot(self.sigma_k, np.dot(self.M.T, np.linalg.inv(aux_matrix_C)))

        self.x = np.round(self.x + np.dot(kalman_gain, (self.z - np.dot(self.M, self.x))))
        self.sigma_k = self.sigma_k - np.dot(kalman_gain, np.dot(aux_matrix_C, kalman_gain.T))
        self.last_x = self.x
        return self.x
