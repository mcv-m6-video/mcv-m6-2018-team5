
class KalmanFilter:

    def __init__(self, initialPosition):

        self.prioriEstimate = initialPosition
        self.posterioriError = [0, 0]
        self.prioriError = [0, 0]
        self.gain = [0, 0]
        self.Q = [1e-5, 1e-5]  # process variance
        self.R = [0.1 ** 2, 0.1 ** 2]  # estimate of measurement variance


    def predictKalmanFilter(self):
        return self.prioriEstimate

    def updateMeasurement(self, currentPosition):
        self.prioriError = self.posterioriError + self.Q
        self.gain = self.prioriError / (self.prioriError + self.R)
        self.prioriEstimate = self.prioriEstimate + self.gain * (currentPosition - self.prioriEstimate)
        self.posterioriError = (1-self.gain) * self.prioriError
