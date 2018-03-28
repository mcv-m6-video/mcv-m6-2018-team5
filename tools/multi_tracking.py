import cv2 as cv
import numpy as np
from kalman_filter_2 import KalmanFilter

nextId = 1

class Track(object):

    def __init__(self, id, bbox, kalmanFilter):
        self.id = id
        self.bbox = bbox
        self.kalmanFilter = kalmanFilter
        self.age = 1
        self.totalVisibleCount = 1
        self.consecutiveInvisibleCount = 0

def predictNewLocationsOfTracks(tracks):
    for track in tracks:
        bbox = track.bbox

        # Predict the current location of the track.
        predictedCentroid = track.kalmanFilter.predict()

        # Shift the bounding box so that its center is at the predicted location.
        predictedX = predictedCentroid[0] - bbox[2] / 2
        predictedY = predictedCentroid[1] - bbox[3] / 2
        track.bbox = (predictedX, predictedY, bbox[2], bbox[3])


def detectionToTrackAssignment(tracks, centroids):
    numTracks = len(tracks)

    # Compute the cost of assigning each detection to each track.
    costs = []
    for i in range(numTracks):
        costs.append(tracks[i].kalmanFilter.distance(centroids))

    # Solve the assignment problem.
    costOfNonAssignment = 20
    assignments = [np.argmin(cost) if(np.amin(cost) < costOfNonAssignment) else -1 for cost in costs]

    # Identify tracks with no assignment, if any
    unassignedTracks = []
    for i in range(len(assignments)):
        tracks[i].age += 1
        if assignments[i] == -1:
            unassignedTracks.append(i)

    # Now look for un_assigned detects
    unassignedDetections = []
    for i in range(len(centroids)):
        if i not in assignments:
            unassignedDetections.append(i)

    return assignments, unassignedTracks, unassignedDetections


def updateAssignedTracks(tracks, bboxes, centroids, assignments):
    numAssignedTracks = len(assignments)
    for i in range(numAssignedTracks):
        if assignments[i] != -1:
            trackIdx = i
            detectionIdx = assignments[i]
            centroid = centroids[detectionIdx]
            bbox = bboxes[detectionIdx]

            # Correct the estimate of the object's location using the new detection.
            tracks[trackIdx].kalmanFilter.update(centroid)

            # Replace predicted bounding box with detected bounding box.
            tracks[trackIdx].bbox = bbox

            # Update track's age.
            tracks[trackIdx].age += 1

            # Update visibility.
            tracks[trackIdx].totalVisibleCount += 1
            tracks[trackIdx].consecutiveInvisibleCount = 0


def updateUnassignedTracks(tracks, unassignedTracks):
    for ind in unassignedTracks:
        tracks[ind].age += 1
        tracks[ind].consecutiveInvisibleCount += 1


def deleteLostTracks(tracks):
    if tracks == list():
        return

    invisibleForTooLong = 20
    ageThreshold = 8

    # Compute the fraction of the track's age for which it was visible.
    ages = np.asarray([track.age for track in tracks])
    totalVisibleCounts = np.asarray([track.totalVisibleCount for track in tracks])
    visibility = totalVisibleCounts / ages

    # Find the indices of 'lost' tracks.
    i = 0
    while i < len(tracks):
        if (ages[i] < ageThreshold and visibility[i] < 0.6) or tracks[i].consecutiveInvisibleCount >= invisibleForTooLong:
            del tracks[i]
        else:
            i += 1


def createNewTracks(tracks, bboxes, centroids, unassignedDetections):
    unassignedCentroids = [centroids[i] for i in range(len(centroids)) if (i in unassignedDetections)]
    unassignedBboxes = [bboxes[i] for i in range(len(bboxes)) if i in unassignedDetections]

    for (centroid, bbox) in zip(unassignedCentroids, unassignedBboxes):
        # Create a Kalman filter object.
        kalmanFilter = KalmanFilter(centroid)

        # Create a new track.
        newTrack = Track(nextId, bbox, kalmanFilter)

        # Add it to the array of tracks.
        tracks.append(newTrack)

        # Increment the next id.
        global nextId
        nextId = nextId + 1
