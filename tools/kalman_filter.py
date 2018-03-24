class KalmanFilter:

    def __init__(self, initial_position):
        self.priori_estimate = initial_position
        self.posteriori_error = [0., 0.]
        self.priori_error = [0., 0.]
        self.gain = [0., 0.]
        self.Q = [1e-5, 1e-5]  # process variance
        self.R = [0.1 ** 2, 0.1 ** 2]  # estimate of measurement variance

    def predict_kalman_filter(self):
        return self.priori_estimate

    # noinspection PyTypeChecker
    def update_measurement(self, current_position):
        self.priori_error = self.posteriori_error + self.Q
        self.gain = self.priori_error / (self.priori_error + self.R)
        self.priori_estimate = self.priori_estimate + self.gain * (current_position - self.priori_estimate)
        self.posteriori_error = (1 - self.gain) * self.priori_error
