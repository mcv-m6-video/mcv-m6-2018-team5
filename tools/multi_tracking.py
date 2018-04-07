import numpy as np

from kalman_filter import KalmanFilter


class Track(object):

    def __init__(self, idx, bbox, kalman_filter):
        # Kalman filter
        self.positions = [[kalman_filter.x[0], kalman_filter.x[2]]]
        self.predictions = [[kalman_filter.x[0], kalman_filter.x[2]]]
        self.kalman_filter = kalman_filter
        self.age = 1
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0

        # Vehicle
        self.id = idx
        self.bbox = bbox
        self.current_speed = 0
        self.speeds = []
        self.lane = -1


class MultiTracker(object):

    def __init__(self, cost_of_non_assignment, invisible_too_long, min_age_threshold, kalman_init_parameters):
        self.next_id = 1

        # Placeholders
        self.tracks = []
        self.assignments = []
        self.unassigned_detections = []
        self.unassigned_tracks = []

        # Parameters
        self.invisible_too_long = invisible_too_long
        self.min_age_threshold = min_age_threshold
        self.cost_of_non_assignment = cost_of_non_assignment

        # Kalman init parameters
        self.kalman_init_parameters = kalman_init_parameters

    def create_new_tracks(self, bboxes, centroids):
        unassigned_centroids = [centroids[i] for i in range(len(centroids)) if (i in self.unassigned_detections)]
        unassigned_bboxes = [bboxes[i] for i in range(len(bboxes)) if i in self.unassigned_detections]

        for (centroid, bbox) in zip(unassigned_centroids, unassigned_bboxes):
            # Create a Kalman filter object.
            kalman_filter = KalmanFilter(centroid, **self.kalman_init_parameters)

            # Create a new track.
            new_track = Track(self.next_id, bbox, kalman_filter)

            # Add it to the array of tracks.
            self.tracks.append(new_track)

            # Increment the next id.
            self.next_id = self.next_id + 1

    def delete_lost_tracks(self):
        if len(self.tracks) == 0:
            return

        # Compute the fraction of the track's age for which it was visible.
        ages = np.asarray([track.age for track in self.tracks])
        total_visible_counts = np.asarray([track.total_visible_count for track in self.tracks])
        visibility = total_visible_counts.astype('float32') / ages.astype('float32')

        # Find the indices of 'lost' tracks.
        i = 0
        while i < len(self.tracks):
            if (ages[i] < self.min_age_threshold and visibility[i] < 0.6) or \
                    (self.tracks[i].consecutive_invisible_count >= self.invisible_too_long):
                del self.tracks[i]
            else:
                i += 1

    def detection_to_track_assignment(self, centroids):
        num_tracks = len(self.tracks)

        if len(centroids) != 0:
            # Compute the cost of assigning each detection to each track.
            costs = []
            for i in range(num_tracks):
                costs.append(self.tracks[i].kalman_filter.distance(centroids))

            # Solve the assignment problem.
            assignments = [np.argmin(cost) if (np.amin(cost) < self.cost_of_non_assignment) else -1 for cost in costs]

            # Identify tracks with no assignment, if any
            unassigned_tracks = []
            for i in range(len(assignments)):
                if assignments[i] == -1:
                    unassigned_tracks.append(i)

            # Now look for un_assigned detects
            unassigned_detections = []
            for i in range(len(centroids)):
                if i not in assignments:
                    unassigned_detections.append(i)
        else:
            assignments = [-1] * len(self.tracks)
            unassigned_tracks = range(len(self.tracks))
            unassigned_detections = []

        self.assignments = assignments
        self.unassigned_tracks = unassigned_tracks
        self.unassigned_detections = unassigned_detections

    def predict_new_locations_of_tracks(self):
        for track in self.tracks:
            bbox = track.bbox

            # Predict the current location of the track.
            predicted_pos = track.kalman_filter.predict()
            track.predictions.append([predicted_pos[0], predicted_pos[2]])

            # Shift the bounding box so that its center is at the predicted location.
            predicted_x = predicted_pos[0] - bbox[2] / 2
            predicted_y = predicted_pos[2] - bbox[3] / 2
            track.bbox = (int(predicted_x), int(predicted_y), int(bbox[2]), int(bbox[3]))

    def update_assigned_tracks(self, bboxes, centroids):
        num_assigned_tracks = len(self.assignments)
        for i in range(num_assigned_tracks):
            if self.assignments[i] != -1:
                track_idx = i
                detection_idx = self.assignments[i]
                centroid = centroids[detection_idx]
                bbox = bboxes[detection_idx]

                # Correct the estimate of the object's location using the new detection.
                self.tracks[track_idx].kalman_filter.correct(centroid, True)

                # Replace predicted bounding box with detected bounding box.
                self.tracks[track_idx].bbox = bbox
                position = self.tracks[track_idx].kalman_filter.x
                self.tracks[track_idx].positions.append([position[0], position[2]])

                # Update track's age.
                self.tracks[track_idx].age += 1

                # Update visibility.
                self.tracks[track_idx].total_visible_count += 1
                self.tracks[track_idx].consecutive_invisible_count = 0

    def update_unassigned_tracks(self):
        for ind in self.unassigned_tracks:
            self.tracks[ind].age += 1
            self.tracks[ind].consecutive_invisible_count += 1
            self.tracks[ind].kalman_filter.correct([0, 0], False)
