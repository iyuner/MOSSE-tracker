#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker
from mosse import MOSSETracker
from cvl.features import extract_features, features_to_image
from cvl.features import FEATURES, FEATURES_NAMES

dataset_path = "Mini-OTB"

DEBUG = True
SHOW_TRACKING = True
SEQUENCE_IDX = 5
FEATURE_TYPE = FEATURES.RGB

def fit_size(img, tsize=None, fit_height=True):
    dsize = (int(img.shape[1] * tsize[0] / img.shape[0]), tsize[0]) if fit_height else (tsize[1], int(img.shape[0] * tsize[1] / img.shape[1]))
    img = cv2.resize(img, dsize=dsize)
    return img

def draw_bbox(img, bbox, color=(0, 255, 0)):
    pt0 = (bbox.xpos, bbox.ypos)
    pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
    cv2.rectangle(img, pt0, pt1, color=color, thickness=3)


if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)

    a_seq = dataset[SEQUENCE_IDX]

    if SHOW_TRACKING:
        cv2.namedWindow("tracker")

    # tracker = NCCTracker()
    tracker = MOSSETracker()

    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)}")
        image_color = frame['image']
        image = extract_features(image_color, FEATURE_TYPE)

        if frame_idx == 0:
            bbox = frame['bounding_box']
            if bbox.width % 2 == 0:
                bbox.width += 1

            if bbox.height % 2 == 0:
                bbox.height += 1

            tracker.start(image, bbox)
        else:
            tracker.detect(image)
            tracker.update(image)
        
        if DEBUG:
            N_CHANNELS = 4
            bbox = tracker.region
            gt_bbox = frame['bounding_box']
            image_bgr = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            image_wbox = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            draw_bbox(image_wbox, gt_bbox, color=(0, 255, 0))
            draw_bbox(image_wbox, bbox, color=(255, 0, 0))
            union = gt_bbox.union_box(bbox).area()
            intersection = gt_bbox.intersection_box(bbox).area()
            print(f'iou = {intersection / union:.2f}')
            if tracker.last_response is None:
                image_top = image_wbox
                template = fit_size(tracker.crop_patch(image_bgr), tsize=image_wbox.shape[:2], fit_height=True,)
            else:
                response = features_to_image(tracker.last_response,
                                             cmap=cv2.COLORMAP_JET)
                cv2.circle(response, (tracker.last_loc), radius=2, color=(0,0,0), thickness=1)
                response = fit_size(response, tsize=image_wbox.shape[:2], fit_height=True,)
                patch = fit_size(tracker.crop_patch(image_bgr, tracker.last_region), tsize=image_wbox.shape[:2], fit_height=True,)
                image_top = np.hstack([image_wbox, template, patch, response])
                template = fit_size(tracker.crop_patch(image_bgr), tsize=image_wbox.shape[:2], fit_height=True,)
            feat = np.hstack(image[:,:,:min(image.shape[2], N_CHANNELS)].transpose(2,0,1)) if len(image.shape)==3 else image
            feat = features_to_image(feat)
            feat = fit_size(feat, tsize=image_top.shape[:2], fit_height=False)
            cv2.imshow("tracker", np.vstack([image_top, feat]))
            key = cv2.waitKey(0)
            if key==ord('q'):
                exit()

        elif SHOW_TRACKING:
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            draw_bbox(image_color, tracker.region, color=(0, 255, 0))
            cv2.imshow("tracker", image_color)
            key = cv2.waitKey(0)
            if key==ord('q'):
                exit()
            if key==ord('s'):
                filename = f"./results/{tracker.__class__.__name__}_{FEATURES_NAMES[FEATURE_TYPE]}_s{SEQUENCE_IDX:02}_f{frame_idx:04}.jpg"
                cv2.imwrite(filename, image_color)

