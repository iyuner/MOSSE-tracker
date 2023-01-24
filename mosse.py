import cv2 
import numpy as np

# create mosse tracker

class MOSSETracker():
    
    def __init__(self):
        self._tracker = cv2.TrackerMOSSE_create()
        self._is_initialized = False
        self.path = "../Mini-OTB"
        self.first_img = cv2.imread(self.path + "../Basketball/img/0001.jpg") 
        self.annot_file = self.path + "../anno/Basketball.txt"
        self.n_steps = 1
        self.bounding_boxes = []
        self.get_bb()

    def start(self, image, bbox):
        self._is_initialized = self._tracker.init(image, bbox)

    def detect(self, image):
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized")

        self._is_initialized, bbox = self._tracker.update(image)
        return bbox
    def update(self):
        pass

    def preprocess(self, image:np.array)->np.ndarray:
        # all the pixels are transformed using log function
        image = np.log(image + 1)
        # the image is normalized
        image = (image - np.mean(image)) / np.std(image)
        # the image is multiplied by a cosine window to reduce the effect of the border
        image = image * np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
        return image

    def pretrain_setup(self, first_img):
        # first image is the first frame of the video
        # G is the ground truth bounding box
        # function outputs components of filter H - Ai and Bi
        # cut out the region of interest from the first image 
        x, y, w, h = self.bounding_boxes[0]
        G = first_img[y:y+h, x:x+w]
        # resize first image to the size of the ground truth
        first_img = cv2.resize(first_img, (w, h))
        # normalize image   
        first_img = self.preprocess(first_img)
        # creaete a fourier transform of the image
        first_img = np.fft.fft2(first_img)
        # create a fourier transform of the ground truth
        G = np.fft.fft2(G, s=first_img.shape)
        #  # create a filter H 
        # elementwise multiplication of the fourier transform of the image and the fourier transform of the ground truth
        Ai = G * np.conj(first_img)
        Bi = first_img * np.conj(first_img)
        return Ai, Bi
    
    def get_bb(self):
        # read n_steps lines from the annotation file and save to the list of bounding boxes
        # each line contains 4 numbers - x, y, width, height
        # x, y - coordinates of the top left corner of the bounding box
        # width, height - width and height of the bounding box
        # return the list of bounding boxes
        self.bounding_boxes = []
        with open(self.annot_file, 'r') as f:
            for i in range(self.n_steps):
                line = next(f).strip()
                # separate the numbers in the line and convert them to integers
                line = [int(x) for x in line.split(',')]
                self.bounding_boxes.append(line)

    def pretrain(self, lr = 0.125):

        Ai, Bi = self.pretrain_setup(self.first_img, self.G)
        
    @property
    def region(self):
        if not self._is_initialized:
            raise RuntimeError("Tracker not initialized")

        return self._tracker.getObjects()[0]

    @property
    def is_initialized(self):
        return self._is_initialized

tracker = mosse.MOSSETracker()
