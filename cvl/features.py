import torch
import torch.nn as nn

import scipy.io
import os

import numpy as np
import cv2
from skimage.feature import hog, daisy
from torchvision import transforms
if torch.__version__ == "1.2.0":
    from torchvision.models.utils import load_state_dict_from_url
else:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

class FEATURES:
    GRAYSCALE = 1
    RGB = 2
    HOG = 3
    DAISY = 4
    ALEXNET = 5
    COLORNAMES = 6

FEATURES_NAMES = {1: 'GRAYSCALE', 2: 'RGB', 3: 'HOG', 4: 'DAISY', 5: 'ALEXNET', 6: 'COLORNAMES'}

COLOR_NAMES = ['black', 'blue', 'brown', 'grey', 'green', 'orange',
               'pink', 'purple', 'red', 'white', 'yellow']
COLOR_RGB = [[0, 0, 0] , [0, 0, 1], [.5, .4, .25] , [.5, .5, .5] , [0, 1, 0] , [1, .8, 0] ,
             [1, .5, 1] ,[1, 0, 1], [1, 0, 0], [1, 1, 1 ] , [ 1, 1, 0 ]]

COLORNAMES_TABLE_PATH = os.path.join(os.path.dirname(__file__), 'colornames_w2c.mat')
COLORNAMES_TABLE = scipy.io.loadmat(COLORNAMES_TABLE_PATH)['w2c']

model = None
transform = transforms.Compose(
    [
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def extract_features(image_color, feature_type):
    if feature_type == FEATURES.GRAYSCALE:
        image = np.sum(image_color, 2) / 3
    elif feature_type == FEATURES.RGB:
        image = image_color
    elif feature_type == FEATURES.HOG:
        fd = hog(image_color,
                 orientations=8,
                 pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1),
                 feature_vector=False,
                 channel_axis=-1,)
        image = fd.reshape(fd.shape[0], fd.shape[1], -1)
        image = cv2.resize(image, dsize=image_color.shape[1::-1])
    elif feature_type == FEATURES.DAISY:
        fd = daisy(np.sum(image_color, 2) / 3,
                    step=1,
                    radius=15,
                    rings=1,
                    histograms=2,
                    orientations=4)
        image = fd.reshape(fd.shape[0], fd.shape[1], -1)
        image = cv2.resize(image, dsize=image_color.shape[1::-1])
    elif feature_type == FEATURES.COLORNAMES:
        image = colornames_image(image_color, mode='probability')
    elif feature_type == FEATURES.ALEXNET:
        global model
        if model==None:
            model = alexnetFeatures(True)#.to(device) # For some reason, gpu inference not working, may have to do with conda env
        image = torch.unsqueeze(transform(image_color),0)
        image = image#.to(device)
        with torch.no_grad():
            image = torch.squeeze(model(image,0)).permute(1,2,0).cpu().numpy()
        image = cv2.resize(image, dsize=image_color.shape[1::-1])
    else:
        raise NotImplementedError
    return image


def features_to_image(feat, cmap=None):
    feat = ((feat - feat.min()) / (feat.max() - feat.min()) * 255).astype(np.uint8)
    if cmap is None:
        feat = cv2.cvtColor(feat, cv2.COLOR_GRAY2RGB)
    else:
        feat = cv2.applyColorMap(feat, cmap)
    return feat

def colornames_image(image, mode='probability'):
    """Apply color names to an image
    Parameters
    --------------
    image : array_like
        The input image array (RxC)
    mode : str
        If 'index' then it returns an image where each element is the corresponding color name label.
        If 'probability', then the returned image has size RxCx11 where the last dimension are the probabilities for each
        color label.
        The corresponding human readable name of each label is found in the `COLOR_NAMES` list.
    Returns
    --------------
    Color names encoded image, as explained by the `mode` parameter.
    """
    image = image.astype('double')
    idx = np.floor(image[..., 0] / 8) + 32 * np.floor(image[..., 1] / 8) + 32 * 32 * np.floor(image[..., 2] / 8)
    m = COLORNAMES_TABLE[idx.astype('int')]

    if mode == 'index':
        return np.argmax(m, 2)
    elif mode == 'probability':
        return m
    else:
        raise ValueError("No such mode: '{}'".format(mode))

"""
    These where taken from the torchvision repository, and modified to return the 
    features instead of the classification score.
"""

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNetFeature(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetFeature, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.conv_indexes = [1,4,7,9,11]
        
    def forward(self, x, conv_index=0):
        x = self.features[:self.conv_indexes[conv_index]](x)
        return x


def alexnetFeatures(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetFeature(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
