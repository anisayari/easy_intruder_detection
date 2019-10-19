class TrackableObject:
    def __init__(self, objectID, centroid, current_area):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.current_area = current_area
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False

    def check_change_area(self,current_area):
        if self.current_area == current_area:
            return False
        else:
            self.current_area = current_area
            return True
