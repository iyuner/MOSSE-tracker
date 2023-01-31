from copy import copy
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.signal.windows import gaussian
from scipy.ndimage import gaussian_filter
import cv2
from itertools import product
from matplotlib import pyplot as plt
from .image_io import crop_patch


class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.last_loc = None
        self.last_region = None
        self.region = None
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
        if len(patch.shape) == 3:
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



class ImprovedMOSSETracker(NCCTracker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Various hyperparameters to tune
        # Gaussian standard deviation for the desired response
        self.peak_sigma_relative = 3 / 100
        self.peak_sigma = 2
        # Extend the search window and padding the filter
        self.pad = False
        # Preprocessing
        self.log_transform = True
        self.log_offset = 1
        self.preprocess_window_type = 'cosine'
        # Augmentations
        self.angles = [-1, 0, 1]
        self.scales = [0.95, 1, 1.05]
        self.s1 = [-0.05, 0, 0.05]
        self.s2 = [-0.05, 0, 0.05]
        self.augment_on_update = False

    def normalize_patch(self, patch):
        # Log transform -- sometimes helps, sometimes doesn't
        if self.log_transform:
            patch = np.log(patch.astype(np.float) + self.log_offset)
        
        # Suppress boundaries
        # with cosine window
        if self.preprocess_window_type=='cosine':
            window = np.outer(np.hanning(patch.shape[0]), np.hanning(patch.shape[1]))
        # or gaussian window
        elif self.preprocess_window_type=='gaussian':
            window_sigma = np.mean(patch.shape[:2])
            window = np.outer(gaussian(patch.shape[0], std=window_sigma), gaussian(patch.shape[1], std=window_sigma))
            window = (window-window.min()) / (window.max()-window.min())
        elif self.preprocess_window_type=='none':
            window = np.ones(patch.shape[:2], dtype=np.float)
        else:
            raise NotImplementedError
        # Multi-channel support
        if len(patch.shape) == 3:
            window = window[:,:,None]
        patch = patch * window
        
        # Normalization
        # Multi-channel support
        if len(patch.shape) == 3:
            mean_val = np.mean(patch,(0,1),keepdims=True)
            stdev_val = np.std(patch,(0,1),keepdims=True)
        else:
            mean_val = np.mean(patch)
            stdev_val = np.std(patch)
        patch = patch - mean_val
        patch = patch / stdev_val
        return patch

    def start(self, image, region):
        self.region = copy(region)
        self.search_window = copy(region)
        self.last_region = copy(region)
        self.region_center = (region.height // 2, region.width // 2)

        # Define template 
        patch = self.crop_patch(image)
        fi = self.normalize_patch(patch)
        if self.pad:
            H, W = self.region.height, self.region.width
            self.search_window.xpos -= W // 2
            self.search_window.ypos -= H // 2
            self.search_window.width += W
            self.search_window.height += H
            pad = [(H // 2, H - H // 2), (W // 2, W - W // 2)]
            fi = np.pad(fi, [*pad, (0,0)], mode='constant', constant_values=0)
        Fi = fft2(fi, axes=(0,1))

        # Define desired peaky response
        if self.peak_sigma is None:
            self.peak_sigma = self.peak_sigma_relative * \
                            np.sqrt(region.width**2 + region.height**2)
        gi = np.outer(gaussian(self.search_window.height, std=self.peak_sigma), gaussian(self.search_window.width, std=self.peak_sigma))
        # Inverse FFT shift
        self.gi = ifftshift(gi)
        # Multi-channel support
        if len(patch.shape) == 3:
            self.gi = np.tile(self.gi[:,:,None],[1,1,patch.shape[2]])
        self.Gi = fft2(self.gi, axes=(0,1))

        # Define MOSSE filter
        self.Ai = self.Gi * np.conj(Fi)
        self.Bi = Fi * np.conj(Fi)

        # Affine augmentations
        for scale, s1, s2, angle in product(self.scales, self.s1, self.s2, self.angles):
            if not(angle == 0 and scale == 1 and s1 == 0 and s2 == 0):
                patch_aug = self._affine(patch, angle=angle, scale=scale, s1=s1, s2=s2)
                # plt.imshow(np.hstack([patch, patch_aug]))#, 'gray')
                # plt.show()
                fi = self.normalize_patch(patch_aug)
                if self.pad:
                    fi = np.pad(fi, [*pad, (0,0)], mode='constant', constant_values=0)
                Fi = fft2(fi, axes=(0,1))
                self.Ai += self.Gi * np.conj(Fi)
                self.Bi += Fi * np.conj(Fi)

    def detect(self, image):
        # Take a search window
        patch = self.crop_patch(image, region=self.search_window)
        fi = self.normalize_patch(patch)
        Fi = fft2(fi, axes=(0,1))

        # Compute MOSSE filter
        Hi_conj = self.Ai / self.Bi

        # Correlate
        responsef = Hi_conj * Fi
        response = ifft2(responsef, axes=(0,1)).real
        if len(patch.shape) == 3:
            response = response.sum(2)

        # FFT shift
        response = fftshift(response)

        # Get offset from the mode (argmax) of the response
        r, c = np.unravel_index(np.argmax(response), response.shape)
        r_offset = r - self.region_center[0] - (self.region.height//2 if self.pad else 0)
        c_offset = c - self.region_center[1] - (self.region.width//2 if self.pad else 0)

        # Keep for visualisation
        self.last_response = response
        self.last_loc = (c, r)
        self.last_region.xpos = self.region.xpos
        self.last_region.ypos = self.region.ypos

        # Apply offset sefely
        self.region.xpos += c_offset
        self.region.ypos += r_offset

        height, width = image.shape[:2]
        self.region.xpos = min(width, max(0,self.region.xpos))
        self.region.ypos = min(height, max(0,self.region.ypos))

        self.search_window.xpos = self.region.xpos
        self.search_window.ypos = self.region.ypos
        if self.pad:
            H, W = self.region.height, self.region.width
            self.search_window.xpos -= W // 2
            self.search_window.ypos -= H // 2

        return self.region
    
    def update(self, image, lr=0.2):
        # New template
        patch = self.crop_patch(image)
        fi = self.normalize_patch(patch)
        if self.pad:
            H, W = self.region.height, self.region.width
            pad = [(H // 2, H - H // 2), (W // 2, W - W // 2)]
            fi = np.pad(fi, [*pad, (0,0)], mode='constant', constant_values=0)
        Fi = fft2(fi, axes=(0,1))

        # New filter
        Ai = self.Gi * np.conj(Fi)
        Bi = Fi * np.conj(Fi)

        # Affine augmentations
        if self.augment_on_update:
            for scale, s1, s2, angle in product(self.scales, self.s1, self.s2, self.angles):
                if not(angle == 0 and scale == 1 and s1 == 0 and s2 == 0):
                    patch_aug = self._affine(patch, angle=angle, scale=scale, s1=s1, s2=s2)
                    # plt.imshow(np.hstack([patch,patch_aug]), 'gray'); 
                    # plt.show()
                    fi = self.normalize_patch(patch_aug)
                    if self.pad:
                        fi = np.pad(fi, [*pad, (0,0)], mode='constant', constant_values=0)
                    Fi = fft2(fi, axes=(0,1))
                    Ai += self.Gi * np.conj(Fi)
                    Bi += Fi * np.conj(Fi)

        # Update MOSSE filter
        self.Ai = lr * Ai + (1 - lr) * self.Ai
        self.Bi = lr * Bi + (1 - lr) * self.Bi
    
    def _affine(self, img, angle=None, scale=None, s1=0, s2=0):
        # add small affine pertubation
        if angle is None:
            angle = (np.random.rand() - 0.5) * 20
        if scale is None:
            scale = np.random.rand() * 0.1 + 0.95
        center = (img.shape[1]//2, img.shape[0]//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        M = rot_mat + np.float32(\
            [[rot_mat[0,1] * s2, rot_mat[0,0] * s1, -s1 * center[0]],
             [rot_mat[1,1] * s2, rot_mat[1,0] * s1, -s2 * center[1]]])
        warp_img = cv2.warpAffine(img, M[:2,:3], (img.shape[1], img.shape[0]), 
        cv2.BORDER_REFLECT)
        return warp_img