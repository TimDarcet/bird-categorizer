#! /bin/env python
# Program to crop images before classifying them
# Reused code from https://www.kaggle.com/suyogdahal/object-detection-with-detectron2-pytorch

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import argparse
import numpy as np
from torch.cuda import is_available

parser = argparse.ArgumentParser(description='RecVis A3 cropper script')
parser.add_argument('--path', type=str, default='bird_dataset', metavar='f',
                    help="Path to folder or image")
parser.add_argument('--mode', type=str, default="folder", metavar='M',
                    help='folder / image')
args = parser.parse_args()


# Loading the default config
cfg = get_cfg()

# Merging config from a YAML file
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))


# Downloading and loading pretrained weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")


# Changing some other configs
if is_available():
    cfg.MODEL.DEVICE = 'cuda'
    print("Using GPU")
else:
    cfg.MODEL.DEVICE = 'cpu'
    print("Using CPU")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01  # setting threshold for this model


# Defining the Predictor
predictor = DefaultPredictor(cfg)

def show_image(im):
    """
    Function to display an image
    
    Args:
        im ([numpy.ndarray])
        height ([int] or None)
        width ([int] or None)
    """
    plt.figure(figsize=(16,10))
    plt.imshow(im)
    plt.axis("off")
    plt.show()

def get_predicted_labels(classes, scores, class_names):
    """
    Function to return the name of predicted classes along with accuracy scores
    
    Args:
        classes (list[int] or None)
        scores (list[float] or None)
        class_names (list[str] or None)
    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
        labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
        return labels
    else:
        return "No object identified"

def get_max_label(classes, scores, class_names):
    label = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
        imax = max(zip(scores, labels))
        return label
    else:
        return None

def detect_image(impath, display=True):
    # Read image
    im = mpimg.imread(impath)
    if len(im.shape) == 2:  # Grayscale image
        im = np.stack([im, im, im], axis=-1)

    # Predicting image
    outputs = predictor(im)


    # Extracting other data from the predicted image
    pred_scores = outputs["instances"].scores
    pred_classes = outputs["instances"].pred_classes
    pred_boxes = outputs["instances"].pred_boxes
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    preds = list(zip((class_names[pc] for pc in pred_classes), pred_scores, pred_boxes))

    # Obtaining a list of predicted class labels using the utility function created earlier
    predicted_labels = get_predicted_labels(pred_classes, pred_scores, class_names)


    # Creating the Visualizer for visualizing the bounding boxes
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    output_im = v.get_image()[:, :, ::-1] # image with bounding box and lables defined


    # Displaying the output
    print(f"Predicted Objects: {predicted_labels}")
    show_image(output_im)
    return outputs

def detect_folder(folder):
    rootdir = Path(folder)
    for f in rootdir.rglob("*.jpg"):
        # Read image
        im = mpimg.imread(f)
        if len(im.shape) == 2:  # Grayscale image
            im = np.stack([im, im, im], axis=-1)
        print(f)
        # Predicting image
        outputs = predictor(im)
        
        # Extracting other data from the predicted image
        pred_scores = outputs["instances"].scores
        pred_classes = outputs["instances"].pred_classes
        pred_boxes = outputs["instances"].pred_boxes
        class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
        preds = list(zip((class_names[pc] for pc in pred_classes), pred_scores, pred_boxes))
        print(f.name, preds)

def det_crop_folder(folder):
    rootdir = Path(folder)
    for f in rootdir.rglob("*.jpg"):
        try:
            # Read image
            im = mpimg.imread(f)
            if len(im.shape) == 2:  # Grayscale image
                im = np.stack([im, im, im], axis=-1)

            # Predicting image
            outputs = predictor(im)
            
            # Extracting other data from the predicted image
            pred_scores = outputs["instances"].scores
            if len(pred_scores) == 0:
                print("what", f)
            pred_classes = outputs["instances"].pred_classes
            pred_boxes = outputs["instances"].pred_boxes
            class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            preds = list(zip((class_names[pc] for pc in pred_classes), pred_scores, pred_boxes))
            # print(f.name, preds)
            minx = miny = float('inf')
            maxx = maxy = -float('inf')
            max_bird_score = -float('inf')
            max_bird_box = None
            for i, (label, score, box) in enumerate(preds):
                if label == "bird":
                    max_bird_score = max(max_bird_score, score)
                    max_bird_box = box
                if label == "bird" and score > 0.5:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    minx = min(minx, x1, x2)
                    miny = min(miny, y1, y2)
                    maxx = max(maxx, x1, x2)
                    maxy = max(maxy, y1, y2)
            if max_bird_box is not None:
                    x1, y1, x2, y2 = max_bird_box.cpu().numpy()
                    minx = min(minx, x1, x2)
                    miny = min(miny, y1, y2)
                    maxx = max(maxx, x1, x2)
                    maxy = max(maxy, y1, y2)
            encompassing_bbox = (minx, miny, maxx, maxy)
            p = ("crops_square" / f)
            p.parent.mkdir(parents=True, exist_ok=True)
            mpimg.imsave(p, crop_square(im, encompassing_bbox))
            print(f"Cropped {f.name} to {minx}, {miny}, {maxx}, {maxy}")
        except Exception as e:
            print(f, e, encompassing_bbox, file=sys.stderr)

def crop(im, box, fact=1.5):
    a, b, c, d = box
    x1, y1, x2, y2 = box
    m = (x1 + x2) / 2, (y1 + y2) / 2
    w = abs(x1 - x2) * fact
    h = abs(y1 - y2) * fact
    def inint(x, mx):
        return int(max(0, min(int(x), int(mx))))
    x1 = inint(m[0] - w / 2, im.shape[1] - 1)
    y1 = inint(m[1] - h / 2, im.shape[0] - 1)
    x2 = inint(m[0] + w / 2, im.shape[1] - 1)
    y2 = inint(m[1] + h / 2, im.shape[0] - 1)
    cropped = im[y1:y2, x1:x2]
    return cropped

def crop_square(im, box, fact=1.5):
    a, b, c, d = box
    x1, y1, x2, y2 = box
    m = (x1 + x2) / 2, (y1 + y2) / 2
    w = abs(x1 - x2) * fact
    h = abs(y1 - y2) * fact
    h = max(w, h)
    w = h
    def inint(x, mx):
        return int(max(0, min(int(x), int(mx))))
    x1 = inint(m[0] - w / 2, im.shape[1] - 1)
    y1 = inint(m[1] - h / 2, im.shape[0] - 1)
    x2 = inint(m[0] + w / 2, im.shape[1] - 1)
    y2 = inint(m[1] + h / 2, im.shape[0] - 1)
    cropped = im[y1:y2, x1:x2]
    return cropped

if args.mode == "folder":
    det_crop_folder(args.path)
elif args.mode == "image":
    out = detect_image(args.path)
else:
    raise Exception()

