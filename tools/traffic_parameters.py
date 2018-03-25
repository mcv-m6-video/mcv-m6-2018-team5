import numpy as np

def speed_estimation(prevPos, curPos, prevFram, curFram, pix2met, fram2sec):
    dist = (np.sqrt(np.sum(np.square(curPos - prevPos)))) * pix2met
    time = (curFram - prevFram) * fram2sec

    speed = dist / time * (1 / 1000) * 3600

    return speed
