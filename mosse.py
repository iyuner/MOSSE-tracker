import cv2 
import numpy as np
from cvl.image_io import crop_patch

class MOSSETracker():
    def __init__(self, learning_rate=0.125):
        self.region = None
        self.lr = learning_rate
        self.init()

    def init(self):
        self.A = 0
        self.B = 0
        self.G = 0 # Gaussian peak in the target
        self.sigma = 2
        self.n_affine = 8

    def crop_patch(self, image):
        region = self.region
        return crop_patch(image, region)
    
    def start(self, image_, region):
        assert len(image_.shape) == 2, "MOSSE is only defined for grayscale images"
        self.init()
        self.region = region
        image = image_ / 255
        self._pretrain(image)

    def detect(self, image_):
        assert len(image_.shape) == 2, "MOSSE is only defined for grayscale images"
        image = image_ / 255  # using image/=255 would be applied both in update and detect!
        H = self.A / self.B
        image = self.crop_patch(image)
        # preprocess the image
        image = self._preprocess(image).astype(np.float32)

        # multiply the filter H by the fourier transform of the image
        G = H * np.fft.fft2(image)
        Gi = np.real(np.fft.ifft2(G))

        #if len(patch.shape) == 3:
        #    Gi = Gi.sum(2)

        r, c = np.unravel_index(np.argmax(Gi), Gi.shape)
        # get the coordinates of the top left corner of the bounding box
        # the coordinates are calculated using the coordinates of the maximum value
        # and the size of the bounding box
        dx = int(np.float(c) - self.region.width / 2)
        dy = int(np.float(r) - self.region.height / 2)
        # create a new bounding box
        self.region.xpos += dx
        self.region.ypos += dy
        return self.region
    
    def update(self, image_):
        image = image_ / 255  # using image/=255 would be applied both in update and detect!
        image = self.crop_patch(image)
        # preprocess the image
        image = self._preprocess(image).astype(np.float32)
        Fi = np.fft.fft2(image)
        self.A = self.lr * (self.G * np.conj(Fi)) + (1 - self.lr) * self.A
        self.B = self.lr * (Fi) * np.conjugate(Fi) + (1 - self.lr) * self.B

    def _pretrain(self, img):
        # clipping the image, cut out the region of interest from the first image 
        img = self.crop_patch(img)
        # get gaussian response
        self.G = np.fft.fft2(self._get_gauss(self.region.width, self.region.height))
        
        fi = self._preprocess(img)
        Fi = np.fft.fft2(fi)
        self.A += self.G * np.conj(Fi)
        self.B += Fi * np.conjugate(Fi)

        # do several small affine pertubations
        for i in range(self.n_affine):
            fi = self._preprocess(self._random_affine(img))
            Fi = np.fft.fft2(fi)
            self.A += self.G * np.conj(Fi)
            self.B += Fi * np.conj(Fi)
    
    def _get_gauss(self, w, h):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        c = np.exp( - ((xs - (w-1)/2)**2 + (ys - (h-1)/2)**2) / (2 * self.sigma**2))
        return c
    
    def _preprocess(self, image:np.ndarray)->np.ndarray:
        # all the pixels are transformed using log function
        image = np.log(image + 1)
        image = (image - np.mean(image)) / (np.std(image) + 1e-5)
        # the image is normalized
        # the image is multiplied by a cosine window to reduce the effect of the border
        image = image * np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
        return image
    
    def _random_affine(self, img):
        # add small affine pertubation
        angle = (np.random.rand() - 0.5) * 20   # random angle from [-10, 10) degree
        scale = np.random.rand() * 0.1 + 0.95   # random scale from [0.95, 1.05)
        center = (img.shape[1]//2, img.shape[0]//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        warp_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), cv2.BORDER_REFLECT)
        return warp_img