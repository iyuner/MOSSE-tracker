import cv2 
import numpy as np
import os, re, sys
from cvl.image_io import crop_patch
# create mosse tracker

class MOSSETracker():
    
    def __init__(self, trail = "Basketball"):
        self.path = "Mini-OTB"
        self.images_path = self.path + f"/{trail}/img"
        self.annot_file = self.path + f"/anno/{trail}.txt"
        self.n_steps = 1
        self.bounding_boxes = []
        self.get_bb()
        self.predicted_bounding_boxes = []
        self.verbose = True
        self.lr = 0.125
        self.A = 0
        self.B = 0
        self.G = 0 # Gaussian peak in the target
        self.sigma = 2
        self.n_affine = 8

    def start(self):
        # start from pretraining on n_steps frames to get filter H
        self.pretrain()
        directory = os.listdir(self.images_path)
        directory.sort()

        # initial bb 
        bb = self.bounding_boxes[self.n_steps - 1]
        for i in range(self.n_steps, len(os.listdir(self.images_path))):
            print(f"step {i}")
            H = self.A / self.B
            # read the next frame
            image_name = directory[i]
            image_ori = cv2.imread(os.path.join(self.images_path, image_name))
            # trun image to grayscale
            image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            # clip the image to the size of the bounding box from the previous frame
            image = self._clip_the_image(image, bb)
            # center = (np.float(bb[0]), np.float(bb[1]))
            # image = cv2.getRectSubPix(image, (bb[2], bb[3]), center)
            # preprocess the image
            image = self.preprocess(image).astype(np.float32)

            # multiply the filter H by the fourier transform of the image
            G = H * np.fft.fft2(image)
            Gi = np.real(np.fft.ifft2(G))
            r, c = np.unravel_index(np.argmax(Gi), Gi.shape)
            # get the coordinates of the top left corner of the bounding box
            # the coordinates are calculated using the coordinates of the maximum value
            # and the size of the bounding box
            dx = int(np.float(c) - bb[2] / 2)
            dy = int(np.float(r) - bb[3] / 2)
            
            # create a new bounding box
            bb = [np.abs(bb[0] + dx), np.abs(bb[1] + dy), bb[2], bb[3]]
            print(bb)
            # save the new bounding box to the list of bounding boxes
            self.predicted_bounding_boxes.append(bb)
            # if verbose is True, show the image with the bounding box
            if self.verbose:
                # image = cv2.imread(os.path.join(self.images_path, image_name))
                cv2.rectangle(image_ori, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 0), 2)
                cv2.imshow("image", image_ori)
                cv2.waitKey(0)

            image = cv2.imread(os.path.join(self.images_path, image_name))
            # trun image to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            # clip the image to the size of the bounding box from the previous frame
            # center = (np.float(bb[0]), np.float(bb[1]))
            # image = cv2.getRectSubPix(image, (bb[2], bb[3]), center)
            image = self._clip_the_image(image, bb)
            # preprocess the image
            image = self.preprocess(image)
            Fi = np.fft.fft2(image)
            self.A = self.lr * (self.G * np.conj(Fi)) + (1 - self.lr) * self.A
            self.B = self.lr * (Fi) * np.conjugate(Fi) + (1 - self.lr) * self.B

    def _clip_the_image(self, image:np.ndarray, bb:list = None, step:int = None):
        # clip the image to the size of the bounding box from  
        if bb is not None:
            xpos = bb[0]
            ypos = bb[1]
            width = bb[2]
            height = bb[3]

            r0 = ypos
            r1 = ypos + height
            c0 = xpos
            c1 = xpos + width

            ri0 = r0
            ri1 = r1
            rp0 = 0
            rp1 = height

            ci0 = c0
            ci1 = c1
            cp0 = 0
            cp1 = width

            if c0 < 0:
                ci0 = 0
                cp0 = -c0

            if r0 < 0:
                ri0 = 0
                rp0 = -r0

            if r1 >= image.shape[0]:
                ri1 = image.shape[0]
                rp1 = height - (r1 - image.shape[0])

            if c1 >= image.shape[1]:
                ci1 = image.shape[1]
                cp1 = width - (c1 - image.shape[1])

            patch = np.zeros(shape=(height, width),
                            dtype=image.dtype)

            patch[rp0:rp1, cp0:cp1] = image[ri0:ri1, ci0:ci1]

            assert patch.shape == (height, width)

            return patch
        elif step is not None:
            x, y, w, h = self.bounding_boxes[step]
            image = image[y:y+h, x:x+w]
            return image      
        else:
            raise ValueError("You need to specify either bb or step")
    
    def preprocess(self, image:np.ndarray)->np.ndarray:
        # all the pixels are transformed using log function
        image = np.log(image +1)
        image = (image - np.mean(image)) / (np.std(image) + 1e-5)
        # the image is normalized
        # the image is multiplied by a cosine window to reduce the effect of the border
        image = image * np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
        return image

    def pretrain_step(self, i):
        # first image is the first frame of the video
        # G is the ground truth bounding box
        # function outputs components of filter H - Ai and Bi   
        # first_image - read image i from self.images_path
        x, y, w, h = self.bounding_boxes[i]
        directory = os.listdir(self.images_path)
        directory.sort()
        image_name = directory[i]
        img = cv2.imread(os.path.join(self.images_path, image_name))
        if self.verbose:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("image", img)
            cv2.waitKey(0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) /255
        # clipping the image, cut out the region of interest from the first image 
        img = self._clip_the_image(img, step = i)
        # get gaussian response
        self.G = np.fft.fft2(self._get_gauss(w, h))
        
        fi = self.preprocess(img)
        Fi = np.fft.fft2(fi)
        self.A += self.G * np.conj(Fi)
        self.B += Fi * np.conjugate(Fi)

        # do several small affine pertubations
        for i in range(self.n_affine):
            fi = self.preprocess(self._random_affine(img))
            Fi = np.fft.fft2(fi)
            self.A += self.G * np.conj(Fi)
            self.B += Fi * np.conj(Fi)

    def pretrain(self):
        # pretrain the tracker using the first n_steps frames of the video
        for i in range(self.n_steps):
            self.pretrain_step(i)
        
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
                try:
                    line = [int(x) for x in line.split(',')]
                except:
                    line = [int(x) for x in re.split(r'\t+', line.rstrip('\t'))]
                print(line)
                self.bounding_boxes.append(line)
    
    def _get_gauss(self, w, h):
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        c = np.exp( - ((xs - w/2)**2 + (ys - h/2)**2) / (2 * self.sigma**2) )
        return c

    def _random_affine(self, img):
        # add small affine pertubation
        angle = (np.random.rand() - 0.5) * 20   # random angle from [-10, 10) degree
        scale = np.random.rand() * 0.2 + 0.9   # random scale from [0.9, 1.1)
        center = (img.shape[1]//2, img.shape[0]//2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
        warp_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), cv2.BORDER_REFLECT)
        return warp_img

if __name__ == "__main__":
    try:
        trail = sys.argv[1]
        tracker = MOSSETracker(trail)
    except:
        tracker = MOSSETracker()
    tracker.start()