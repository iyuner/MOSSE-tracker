import cv2 
import numpy as np

# create mosse tracker

class MOSSETracker():
    
    def __init__(self):
        self._tracker = cv2.TrackerMOSSE_create()
        self._is_initialized = False
        self.path = "../Mini-OTB"
    def start(self, image, bbox):
        self._is_initialized = self._tracker.init(image, bbox)

    def detect(self, image):
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized")

        self._is_initialized, bbox = self._tracker.update(image)
        return bbox
    def update(self):
        pass

    def preprocess(self, image):
        # all the pixels are transformed using log function
        image = np.log(image + 1)
        # the image is normalized
        image = (image - np.mean(image)) / np.std(image)
        # the image is multiplied by a cosine window to reduce the effect of the border
        image = image * np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
        return image
        

    @property
    def region(self):
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized")

        return self._tracker.getObjects()[0]

    @property
    def is_initialized(self):
        return self._is_initialized

tracker = mosse.MOSSETracker()
