from copy import copy
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch


class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.last_loc = None
        self.last_region = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def crop_patch(self, image, region=None):
        region = self.region if region is None else region
        if len(image.shape) == 2:
            patch = crop_patch(image, region)
        else:
            patch = []
            for i in range(image.shape[2]):
                patch.append(crop_patch(image[:,:,i], region))
            patch = np.dstack(patch)
        return patch

    def normalize_patch(self, patch):
        patch = patch/255
        if len(patch.shape) == 2:
            mean_val = np.mean(patch,(0,1),keepdims=True)
            stdev_val = np.std(patch,(0,1),keepdims=True)
        else:
            mean_val = np.mean(patch)
            stdev_val = np.std(patch)
        patch = patch - mean_val
        patch = patch / stdev_val
        return patch

    def start(self, image, region):
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.last_region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)
        patch = self.normalize_patch(patch)
        
        self.template = fft2(patch, axes=(0,1))

    def detect(self, image):
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = self.normalize_patch(patch)
        patchf = fft2(patch, axes=(0,1))

        responsef = np.conj(self.template) * patchf
        response = ifft2(responsef, axes=(0,1)).real
        if len(patch.shape) == 3:
            response = response.sum(2)

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response
        self.last_loc = (c, r)
        self.last_region.xpos = self.region.xpos
        self.last_region.ypos = self.region.ypos

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        height, width = image.shape[:2]
        self.region.xpos = min(width, max(0,self.region.xpos))
        self.region.ypos = min(height, max(0,self.region.ypos))

        return self.region

    def update(self, image, lr=0.1):
        # assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = self.normalize_patch(patch)
        patchf = fft2(patch, axes=(0,1))
        self.template = self.template * (1 - lr) + patchf * lr
