#!/usr/bin/env python3

import numpy as np
import cv2
import os
import argparse

def prompt():
    parser = argparse.ArgumentParser(description="Image classification with OpenCV DNN")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--model", required=True, help="Path to pre-trained model weights")
    parser.add_argument("--config", required=True, help="Path to model configuration file (prototxt)")
    parser.add_argument("--labels", required=True, help="Path to class labels file")
    parser.add_argument("--backend", default="opencv", help="Computation backend to use for DNN")
    args = parser.parse_args()
    return args

def load_image(image) -> np.ndarray:
    img = cv2.imread(image)
    if img is None:
        raise ValueError(f"Error: Failed to read and load image file - {image}")
    return img

def load_classes(labels) -> list:
    with open(labels, "r") as fh:
        labels = [line.strip().split(" ", 1)[1] for line in fh.readlines()]
    return labels

def main():
    try:
        args = prompt()

        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Error: Image file does not exist - {args.image}")
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Error: Model file does not exist - {args.model}")
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Error: Config file does not exist - {args.config}")
        if not os.path.exists(args.labels):
            raise FileNotFoundError(f"Error: Labels file does not exist - {args.labels}")

        image_ndarray = load_image(args.image)
        print(f"Image type: {type(image_ndarray)} and shape: {image_ndarray.shape}")

        classes = load_classes(args.labels)
        print(f"Classes: {classes}")

    except FileNotFoundError as e:
        print(f"[File Error]: {str(e)}")
    except ValueError as e:
        print(f"[Value Error]: {str(e)}")
    except Exception as e:
        print(f"[Unexpected Error]: {str(e)}")

if __name__ == "__main__":
    main()