#!/usr/bin/env python3

import numpy as np
import cv2
import os
import argparse
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

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

        # load image
        image_ndarray = load_image(args.image)
        logging.debug(f"Image type: {type(image_ndarray)} and shape: {image_ndarray.shape}")

        # load class labels
        classes = load_classes(args.labels)
        logging.debug(f"Classes: {classes}")

        # load pre-trained model
        net = cv2.dnn.readNetFromCaffe(args.config, args.model)

        # pre-process: convert to blob
        blob = cv2.dnn.blobFromImage(image_ndarray, scalefactor=1.0, size=(224, 224), mean=(104, 117, 123))

        # perform inference
        net.setInput(blob)

        # forward pass
        start_time = time.time()
        output = net.forward()
        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Elapsed time: {elapsed_time:.4f} seconds")

        # print top prediction
        idx = np.argsort(output[0])[::-1][:1]
        logging.info(f"Top prediction: {classes[idx[0]]} with probability: {output[0][idx[0]]:.2f}")

        # display image
        cv2.imshow("Image", image_ndarray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"[File Error]: {str(e)}")
    except ValueError as e:
        print(f"[Value Error]: {str(e)}")
    except Exception as e:
        print(f"[Unexpected Error]: {str(e)}")

if __name__ == "__main__":
    main()