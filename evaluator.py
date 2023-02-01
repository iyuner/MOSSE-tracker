#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from copy import copy
from sklearn.metrics import precision_recall_curve, average_precision_score

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker, ImprovedMOSSETracker
from mosse import MOSSETracker
from cvl.features import FEATURES, FEATURES_NAMES, extract_features


dataset_path = "Mini-OTB"

SAVE_AP_MIOU_TABLE = True
SHOW_PER_SEQUENCE_AUC = True
SHOW_FULL_AUC = False
REEVAL = False
TRACKER_LIST = [NCCTracker(), ImprovedMOSSETracker()]#, MOSSETracker()] 

FEATURES_LIST = [FEATURES.GRAYSCALE, FEATURES.RGB]#, FEATURES.COLORNAMES, FEATURES.HOG, FEATURES.DAISY,FEATURES.ALEXNET]

def evaluate_method_on_sequence(tracker, feature_type, dataset, sequence_idx):
    # print(f"{sequence_idx}: {dataset.sequence_names[sequence_idx]}")
    method = f"{tracker.__class__.__name__}_{FEATURES_NAMES[feature_type]}"
    filename = f"cached/{method}/{dataset.sequence_names[sequence_idx]}.npy"
    if not os.path.exists(filename) or REEVAL:
        a_seq = dataset[sequence_idx]
        track_output = []
        for frame_idx, frame in enumerate(a_seq):
            print(f"{frame_idx} / {len(a_seq)}")
            image_color = frame['image']
            image = extract_features(image_color, feature_type)
            if frame_idx == 0:
                bbox = copy(frame['bounding_box'])
                if bbox.width % 2 == 0:
                    bbox.width += 1
                if bbox.height % 2 == 0:
                    bbox.height += 1
                tracker.start(image, copy(bbox))
            else:
                tracker.detect(image)
                tracker.update(image)
            track_output.append(copy(tracker.region))
        iou = np.array(dataset.calculate_per_frame_iou(sequence_idx, track_output))
        if not os.path.exists(f"cached/{method}"):
            os.makedirs(f"cached/{method}")
        np.save(filename, iou)
    else:
        iou = np.load(filename)
    auc = np.cumsum(iou)
    return auc, iou

if __name__ == "__main__":
    
    dataset = OnlineTrackingBenchmark(dataset_path)

    ### AP & mIoU Table
    if SAVE_AP_MIOU_TABLE:
        for tracker in TRACKER_LIST:
            for feature_type in FEATURES_LIST:
                tracker_id = f"{tracker.__class__.__name__}_{FEATURES_NAMES[feature_type]}"
                filename = f"./results/{tracker_id}.txt"
                data = []
                for sequence_idx in range(len(dataset.sequences)):
                    auc, iou = evaluate_method_on_sequence(tracker, feature_type,dataset, sequence_idx)
                    AP = 0
                    iou_thresholds = np.arange(0,1,0.05)
                    for threshold in iou_thresholds:
                        AP += sum(iou >= threshold) / iou.shape[0]
                    AP /= iou_thresholds.shape[0]
                    mIOU = iou.mean()
                    print(f'{sequence_idx}: Average precision: {AP*100:.2f}%, Mean IoU: {mIOU*100:.2f}%, # Frames: {len(dataset[sequence_idx])}')
                    data.append([sequence_idx, len(dataset[sequence_idx]), mIOU*100, AP*100])
                    np.savetxt(filename, np.array(data), fmt=['%2d', '%5.2f','%5.2f','%d'], delimiter=' & ')
            A = np.array(data)
            B = A.mean(0)
            B[0] = -1
            B[1] = -1
            np.savetxt(filename, np.vstack([A,B]), fmt=['%2d', '%4d', '%5.2f','%5.2f'], delimiter=' & ')

    ### Per-sequence AUC curve
    if SHOW_PER_SEQUENCE_AUC:
        for sequence_idx in range(len(dataset.sequences)):
            for tracker in TRACKER_LIST:
                for feature_type in FEATURES_LIST:
                    auc, iou = evaluate_method_on_sequence(tracker, feature_type,dataset, sequence_idx)
                    label = f"{tracker.__class__.__name__} + {FEATURES_NAMES[feature_type]}"
                    plt.plot(auc/len(dataset[sequence_idx]), '--', label=label)
                    plt.title(f'#{sequence_idx}: {dataset.sequence_names[sequence_idx]} ({len(dataset[sequence_idx])} frames)')
            plt.xlabel('Frame')
            plt.ylabel('AUC')
            plt.legend()
            plt.show()

    ### Full AUC curve
    if SHOW_FULL_AUC:
        for tracker in TRACKER_LIST:
            for feature_type in FEATURES_LIST:
                iou = []
                for sequence_idx in range(len(dataset.sequences)):
                    _, iou_seq = evaluate_method_on_sequence(tracker, feature_type,
                                                    dataset, sequence_idx)
                    iou.append(iou_seq)
                auc = np.cumsum(np.hstack(iou))
                label = f"{tracker.__class__.__name__} + {FEATURES_NAMES[feature_type]}"
                total_length = sum([len(d) for d in dataset.sequences])
                plt.plot(auc/total_length, '--', label=label)
                plt.title(f'Mini-OTB ({total_length} frames)')
        plt.xlabel('Frame')
        plt.ylabel('AUC')
        plt.legend()
        plt.show()