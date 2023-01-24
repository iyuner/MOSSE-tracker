import cv2 
import numpy as np
import os, re, sys
# create mosse tracker

class MOSSETracker():
    
    def __init__(self, trail = "Basketball"):
        self.path = "../Mini-OTB"
        self.images_path = self.path + f"/{trail}/img"
        self.annot_file = self.path + f"/anno/{trail}.txt"
        self.n_steps = 1
        self.bounding_boxes = []
        self.get_bb()
        self.predicted_bounding_boxes = []
        self.verbose = True

    def start(self):
        # start from pretraining on n_steps frames to get filter H
        Ai, Bi = self.pretrain()
        directory = os.listdir(self.images_path)
        directory.sort()

        # initial bb 
        bb = self.bounding_boxes[self.n_steps - 1]
        for i in range(self.n_steps, len(os.listdir(self.images_path))):
            print(f"step {i}")
            H = Ai / Bi
            # read the next frame
            image_name = directory[i]
            image = cv2.imread(os.path.join(self.images_path, image_name))
            # preprocess the image
            image = self.preprocess(image)
            # clip the image to the size of the bounding box from the previous frame
            image = self.clip_the_image(image, bb)
            image = cv2.resize(image, (bb[2], bb[3]))

            # multiply the filter H by the fourier transform of the image
            G = H * np.fft.fft2(image)
            
            G = (np.fft.ifft2(G))
            # find the maximum value in the response
            max_val = np.max(G)
            # find the coordinates of the maximum value
            max_val_pos = np.where(G == max_val)
            # get the coordinates of the top left corner of the bounding box
            # the coordinates are calculated using the coordinates of the maximum value
            # and the size of the bounding box
            print(max_val_pos)
            dx = max_val_pos[1][0] - bb[2] // 2
            dy = max_val_pos[0][0] - bb[3] // 2
            
            """
            response = np.fft.ifft2(G)
            r, c = np.unravel_index(np.argmax(response), response.shape)

            # # Keep for visualisation
            # self.last_response = response

            dx = np.mod(r + bb[2] // 2, bb[2]) - bb[2] // 2
            dy = np.mod(c + bb[3] // 2, bb[3]) - bb[3] // 2
            """
            # create a new bounding box
            bb = [np.abs(bb[0] + dx), np.abs(bb[1] + dy), bb[2], bb[3]]
            print(bb)
            # save the new bounding box to the list of bounding boxes
            self.predicted_bounding_boxes.append(bb)
            # if verbose is True, show the image with the bounding box
            if self.verbose:
                image = cv2.imread(os.path.join(self.images_path, image_name))
                cv2.rectangle(image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 0), 2)
                cv2.imshow("image", image)
                cv2.waitKey(0)
            # update the filter H
            # Ai, Bi = self.update_step(image, Ai, Bi, i)

    def clip_the_image(self, image:np.ndarray, bb:list = None, step:int = None):
        # clip the image to the size of the bounding box from  
        if bb is not None:
            image = image[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
            return image
        elif step is not None:
            x, y, w, h = self.bounding_boxes[step]
            image = image[y:y+h, x:x+w]
            return image      
        else:
            raise ValueError("You need to specify either bb or step")
    
    def linear_mapping(self, img):
        # linear mapping of the image
        return (img - img.min()) / (img.max() - img.min())

    def preprocess(self, image:np.ndarray)->np.ndarray:
        # all the pixels are transformed using log function
        # trun image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image = np.log(image +1)
        image = (image - np.mean(image)) / (np.std(image) + 1e-5)
        print(np.min(image))
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
        first_img = cv2.imread(os.path.join(self.images_path, image_name))
        if self.verbose:
            cv2.rectangle(first_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("image", first_img)
            cv2.waitKey(0)
        # preprocess the image
        first_img = self.preprocess(first_img)
        # clipping the image 
        # cut out the region of interest from the first image 
        G = first_img[y:y+h, x:x+w]
        # get gaussian response
        # G = self._get_gauss_response(self, img, gt)
        # first_img = self.clip_the_image(first_img, step = i)
        # resize first image to the size of the ground truth
        first_img = cv2.resize(first_img, (w, h))
        # normalize image   
        # first_img = self.preprocess(first_img)
        # creaete a fourier transform of the image
        first_img = np.fft.fft2(first_img)
        # create a fourier transform of the ground truth
        G = np.fft.fft2(G) # , s=first_img.shape
        #  # create a filter H 
        # elementwise multiplication of the fourier transform of the image and the fourier transform of the ground truth
        top = G * np.conj(first_img)
        bottom = first_img * np.conj(first_img)
        return top, bottom


    def _get_gauss_response(self, img, gt):
        # get the shape of the image..
        height, width = img.shape
        # get the mesh grid...
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        # get the center of the object...
        center_x = gt[0] + 0.5 * gt[2]
        center_y = gt[1] + 0.5 * gt[3]
        # cal the distance...
        dist = (np.square(xx - center_x) + np.square(yy - center_y)) / (2 * self.args.sigma)
        # get the response map...
        response = np.exp(-dist)
        # normalize...
        response = self.linear_mapping(response)
        return response

    def pretrain(self, lr = 0.125):
        # pretrain the tracker using the first n_steps frames of the video
        Ai, Bi = 0, 0
        for i in range(self.n_steps):
            top, bottom = self.pretrain_step(i)
            Ai = lr * top + (1 - lr) * Ai
            Bi = lr * bottom + (1 - lr) * Bi
        # return the filter H
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
                try:
                    line = [int(x) for x in line.split(',')]
                except:
                    line = [int(x) for x in re.split(r'\t+', line.rstrip('\t'))]
                print(line)
                self.bounding_boxes.append(line)

if __name__ == "__main__":
    try:
        trail = sys.argv[1]
        tracker = MOSSETracker(trail)
    except:
        tracker = MOSSETracker()
    tracker.start()