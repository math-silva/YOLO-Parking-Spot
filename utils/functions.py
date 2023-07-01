# utils/functions.py

# pylint: disable=unsubscriptable-object
# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import cv2
import os
import shutil
import argparse

from sklearn.metrics import confusion_matrix
from PIL import Image

# ROOT DIRECTORY
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

YOLOV5_VERSIONS = [
    "yolov5n.pt",
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",
    "yolov5x.pt",
    "yolov5n6.pt",
    "yolov5s6.pt",
    "yolov5m6.pt",
    "yolov5l6.pt",
    "yolov5x6.pt"
]

YOLOV8_VERSIONS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
]

# Function to check if a car is occupying a parking spot
def is_occupied(image, annotations, left, top, right, bottom, threshold):
    # Get the car bounding boxes
    car_boxes = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        if class_id == 0:
            car_left = int((x_center - width / 2) * image.shape[1])
            car_top = int((y_center - height / 2) * image.shape[0])
            car_right = int((x_center + width / 2) * image.shape[1])
            car_bottom = int((y_center + height / 2) * image.shape[0])
            car_boxes.append((car_left, car_top, car_right, car_bottom))
    
    # If there are no cars in the image that intersect with the parking spot, return False
    if not car_boxes:
        return False
    
    # Calculate the intersection over union (IoU) between the parking spot and each car
    ious = []
    for car_box in car_boxes:
        intersection_left = max(left, car_box[0])
        intersection_top = max(top, car_box[1])
        intersection_right = min(right, car_box[2])
        intersection_bottom = min(bottom, car_box[3])
        intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)
        parking_spot_area = (right - left) * (bottom - top)
        car_area = (car_box[2] - car_box[0]) * (car_box[3] - car_box[1])
        iou = intersection_area / (parking_spot_area + car_area - intersection_area)
        ious.append(iou)
    
    # Return True if the maximum IoU is above the threshold, False otherwise
    return np.max(ious) > threshold

# Function to draw bounding boxes on the image
def draw_bounding_boxes(image_path, annotation_path, output_path, threshold, highlighted_cars):
    # Load the image
    image = cv2.imread(image_path)
    
    # Read the annotation file
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()
    
    # Create a dictionary to store the class counts
    class_counts = {}

    # Initialize the counts
    occupied_disabled_spot = 0
    empty_disabled_spot = 0
    occupied_spot = 0
    empty_spot = 0
    cars_in_transit = 0
    
    # Process each annotation and store rectangle information in a list
    rectangles = []
    for annotation in annotations:
        # Parse the annotation values
        class_id, x_center, y_center, width, height = map(float, annotation.split())

        # Calculate the bounding box coordinates
        left = int((x_center - width / 2) * image.shape[1])
        top = int((y_center - height / 2) * image.shape[0])
        right = int((x_center + width / 2) * image.shape[1])
        bottom = int((y_center + height / 2) * image.shape[0])


        # Increment the class count
        class_counts[class_id] = class_counts.get(class_id, 0) + 1

        color = (0, 0, 0)
        # Set the color based on the class ID
        if class_id == 0: # Car
            if highlighted_cars:
                color = (0, 165, 255)  # Orange color
            else:
                continue
        elif class_id == 1: # Disabled parking spot
            # Check if there is a car occupying of the parking spot
            if is_occupied(image, annotations, left, top, right, bottom, threshold):
                occupied_disabled_spot += 1
                color = (0, 0, 255)  # Red color
            else:
                color = (255, 0, 0)  # Blue color
        elif class_id == 2: # Parking spot
            # Check if there is a car occupying the parking spot
            if is_occupied(image, annotations, left, top, right, bottom, threshold):
                occupied_spot += 1
                color = (0, 0, 255)  # Red color
            else:
                color = (0, 255, 0)  # Green color

        # Store the rectangle information in the list
        rectangles.append((left, top, right, bottom, color))

    # Sort the rectangles list based on class_id, so that red rectangles are processed last
    rectangles.sort(key=lambda x: x[4] == (0, 0, 255))

    # Draw the bounding box rectangles on the image
    for rectangle in rectangles:
        left, top, right, bottom, color = rectangle
        cv2.rectangle(image, (left, top), (right, bottom), color, 2) # type: ignore

    # Empty parking spots count
    empty_disabled_spot = class_counts.get(1, 0) - occupied_disabled_spot
    empty_spot = class_counts.get(2, 0) - occupied_spot

    # Calculate the total number of cars in transit or parked in non-parking spots
    cars_in_transit = class_counts.get(0, 0) - occupied_disabled_spot - occupied_spot

    alpha = 0.4  # Transparency factor.
    
    # Image legend
    text_position = (image.shape[1] - 350, 30)  # Top-right corner position
    overlay = image.copy()
    cv2.rectangle(overlay, (text_position[0] - 10, text_position[1] - 30), (text_position[0] + 320, text_position[1] + 80), (0, 0, 0), cv2.FILLED) # type: ignore
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) # Add the overlay to the image
    
    text = f'Disabled parking spots: {class_counts.get(1, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate
    
    text = f'Parking spots: {class_counts.get(2, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Cars: {class_counts.get(0, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    text_position = (30, 30)  # Top-left corner position
    overlay = image.copy()
    cv2.rectangle(overlay, (text_position[0] - 10, text_position[1] - 30), (text_position[0] + 585, text_position[1] + 140), (0, 0, 0, 0.8), cv2.FILLED) # type: ignore
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) # Add the overlay to the image

    text = f'Empty disabled parking spots: {empty_disabled_spot}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Occupied disabled parking spots: {occupied_disabled_spot}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Empty parking spots: {empty_spot}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Occupied parking spots: {occupied_spot}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Cars in transit or parked in non-parking spots: {cars_in_transit}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Save the image with bounding boxes
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        'Image File': [os.path.basename(image_path)],
        'Disabled parking spots': [class_counts.get(1, 0)],
        'Parking spots': [class_counts.get(2, 0)],
        'Cars': [class_counts.get(0, 0)],
        'Empty disabled parking spots': [empty_disabled_spot],
        'Occupied disabled parking spots': [occupied_disabled_spot],
        'Empty parking spots': [empty_spot],
        'Occupied parking spots': [occupied_spot],
        'Cars in transit or parked in non-parking spots': [cars_in_transit]
    })
        
    return results

def process_labels(data_path, new_labels_folder):
    # Copy data/labels to .temp/labels
    # Create .temp/labels if it doesn't exist
    temp_labels_path = os.path.join(ROOT, '.temp/labels/')

    # if exists delete it
    if os.path.exists(temp_labels_path):
        shutil.rmtree(temp_labels_path)

    # Copy the labels directory to .temp/labels/
    src_labels_path = os.path.join(data_path, 'labels/')
    shutil.copytree(src_labels_path, temp_labels_path)

    # Process each annotation file in the .temp/labels/ folder
    for annotation_file in os.listdir(temp_labels_path):
        with open(os.path.join(temp_labels_path, annotation_file), 'r+') as f:
            lines = f.readlines()
            # remove all lines that start with 0
            f.seek(0)
            for line in lines:
                if not line.startswith('0'):
                    f.write(line)

            try:
                with open(os.path.join(new_labels_folder, annotation_file), 'r') as new_f:
                    f.write(new_f.read())
            except FileNotFoundError:
                pass

            f.truncate()  # Remove extra lines, if any

    return temp_labels_path



# Function to process all images in a folder
def process_images(data_path: str, output_folder: str, threshold: float = 0.4, highlighted_cars: bool = True, model: str = ''):
    processed_images = 0

    images_folder = os.path.join(data_path, 'images/')
    
    # Check if the model name is provided
    if model != '':
        # If model is a path
        if model.__contains__("/") or model.__contains__("\\"):
            new_labels_folder = model
        else: # if model is a name
            new_labels_folder = os.path.join(ROOT, f'results/{model}/labels/')

        try: 
            print(f'Using labels from {new_labels_folder}')
            labels_folder = process_labels(data_path, new_labels_folder) # process labels
        except FileNotFoundError: # if labels not found
            print(
                f'No labels found for model {model}!\n' +
                'Make sure you wrote the correct model name. Otherwise, train the model first.')
            return pd.DataFrame()
    
    else:
        labels_folder = os.path.join(data_path, 'labels/')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_images_folder = os.path.join(output_folder, 'images/')
    # Create the output images folder if it doesn't exist
    if not os.path.exists(os.path.join(output_folder, 'images/')):
        os.makedirs(os.path.join(output_folder, 'images/'))

    # Create an empty dataframe
    results_df = pd.DataFrame()

    # Process each image and annotation in the folder
    print('Processing images...')
    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        
        # Get the annotation file path in the ../labels/ folder
        annotation_file = os.path.splitext(image_file)[0] + '.txt'
        annotation_path = os.path.join(labels_folder, annotation_file)
        
        # Call the function to draw bounding boxes and save the resulting image
        results = draw_bounding_boxes(image_path, annotation_path, output_images_folder, threshold, highlighted_cars)
        processed_images += 1

        # Append the results to the dataframe, remember that the results is a Series object
        results_df = pd.concat([results_df, results], ignore_index=True)

    print(f'Processed {processed_images} images ✅')
    results_df.to_csv(output_folder + 'output.csv', index=False)  # Set index=False to exclude row numbers

    return results_df


# Function to check if model is a custom model or a pre-trained model
def is_custom_model(model: str, yoloversion: str):
    if not model.endswith(".pt"): # If it doesn't end with .pt, add it
        model = model + ".pt"

    versions = YOLOV8_VERSIONS if yoloversion == "8" else YOLOV5_VERSIONS

    # Check if it's a custom model
    if not model in versions:
        # If model is a path
        if model.__contains__("/") or model.__contains__("\\"):
            model_path = model 
            model = model.split("/")[-1]
            model = model.split("\\")[-1]
        # If model is a name, add a path
        else: 
            model_path = os.path.join(ROOT, f"models/{model}")

    # If it is'nt a custom model, don't add the path
    else:
        model_path = model

    return model, model_path


# Function to split the dataset train and val sets
from typing import List

def split_dataset(data_path: str, train_size: float = 0.8):
    # lets put all the train.txt and val.txt info into a list
    full_list = []
    train_list = []
    val_list = []
    
    # Get the names of the files in image folder
    for file in os.listdir(os.path.join(data_path, 'fold_0/images')):
        # Appends the path of the image to the list
        if file.endswith('.jpg'):
            full_list.append('./images/' + file + '\n')

    # Shuffle the list
    np.random.shuffle(full_list)

    # Split the list into train and val lists
    train_size = int(len(full_list) * train_size)
    train_list = full_list[:train_size]
    train_list_rotated = [x.replace('.jpg', '_rotated.jpg') for x in train_list]
    train_list_rotated2 = [x.replace('.jpg', '_rotated2.jpg') for x in train_list]

    val_list = full_list[train_size:]
    val_list_rotated = [x.replace('.jpg', '_rotated.jpg') for x in val_list]
    val_list_rotated2 = [x.replace('.jpg', '_rotated2.jpg') for x in val_list]

    # Write the train.txt file to fold 0
    with open(os.path.join(data_path, 'fold_0/train.txt'), 'w') as f:
        f.writelines(train_list)
        f.writelines(train_list_rotated)
        f.writelines(train_list_rotated2)

    # Write the val.txt file to fold 0
    with open(os.path.join(data_path, 'fold_0/val.txt'), 'w') as f:
        f.writelines(val_list)

    # Write the train.txt file to fold 1
    with open(os.path.join(data_path, 'fold_1/train.txt'), 'w') as f:
        f.writelines(val_list)
        f.writelines(val_list_rotated)
        f.writelines(val_list_rotated2)

    # Write the val.txt file to fold 1
    with open(os.path.join(data_path, 'fold_1/val.txt'), 'w') as f:
        f.writelines(train_list)

    print(f"Dataset split into train and val sets ✅")

def only_car_label(labels_path):
    # Loop over all labels
    for file in os.listdir(labels_path):
        if file.endswith('.txt'):
            with open(os.path.join(labels_path, file), 'r+') as f:
                # I want to delete all lines that do not start with 0
                lines = f.readlines()
                f.seek(0)  # Go back to the beginning of the file
                for line in lines:
                    if line.startswith('0'):
                        f.write(line)

                f.truncate()  # Remove extra lines, if any
                    
    print(f"Only car labels left ✅")


# Rotate images for data augmentation
def rotate_image_and_bboxes(image, bboxes, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Update the coordinates of the bounding boxes
    rotated_bboxes = []
    for bbox in bboxes:
        class_name, cx, cy, bbox_width, bbox_height = bbox

        # Convert to absolute coordinates
        x_min = int((cx - bbox_width / 2) * width)
        y_min = int((cy - bbox_height / 2) * height)
        x_max = int((cx + bbox_width / 2) * width)
        y_max = int((cy + bbox_height / 2) * height)

        # Rotate the coordinates
        rotated_bbox = cv2.transform(np.array([[[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]]]), rotation_matrix)[0]
        x_min_rot, y_min_rot = np.min(rotated_bbox, axis=0)
        x_max_rot, y_max_rot = np.max(rotated_bbox, axis=0)

        # Convert back to relative coordinates
        x_min_rot_rel = x_min_rot / width
        y_min_rot_rel = y_min_rot / height
        x_max_rot_rel = x_max_rot / width
        y_max_rot_rel = y_max_rot / height

        # Calculate the new center
        new_cx = (x_min_rot_rel + x_max_rot_rel) / 2
        new_cy = (y_min_rot_rel + y_max_rot_rel) / 2

        # Calculate the new width and height
        new_width = x_max_rot_rel - x_min_rot_rel
        new_height = y_max_rot_rel - y_min_rot_rel

        rotated_bboxes.append([class_name, new_cx, new_cy, new_width, new_height])

    return rotated_image, rotated_bboxes


def verify_bboxes(bboxes):
    # Verify if the rotated_bbox is inside the image, if not, ignore this bbox
    new_rotated_bboxes = []
    for bbox in bboxes:
        class_name, cx, cy, bbox_width, bbox_height = bbox
        if not (cx >= 0.0 and cx <= 1.0):
            continue
        
        if not (cy >= 0.0 and cy <= 1.0):
            continue

        new_rotated_bboxes.append(bbox)

    return new_rotated_bboxes

# Data augmentation function with rotation
def data_augmentation(data_path, train_txt):
    train_list = []
    with open(os.path.join(data_path, train_txt), 'r') as f:
        train_list = f.readlines()

    images_path = os.path.join(data_path, 'images')
    labels_path = os.path.join(data_path, 'labels')

    # now leave only the last name without \n
    train_list = [(x.split('/')[-1].split('.')[0]) + '.jpg' for x in train_list]
    
    # Loop over all images
    for file in os.listdir(images_path):
        if file not in train_list:
            continue
        if file.endswith('.jpg'):
            # Open image
            image = cv2.imread(os.path.join(images_path, file))

            # Open labels in .txt
            bboxes = []
            with open(os.path.join(labels_path, file.replace('.jpg', '.txt')), 'r') as f:
                lines = f.readlines()
                
                for line in lines:
                    bbox = line.split()
                    bbox = [float(x) for x in bbox]
                    bboxes.append(bbox)

            # Rotate image and update bounding boxes coordinates

            # Rotate 30 degrees
            rotated_image, rotated_bboxes = rotate_image_and_bboxes(image, bboxes, 30)

            # Verify if the rotated_bbox is inside the image, if not, ignore this bbox
            rotated_bboxes = verify_bboxes(rotated_bboxes)

            # Create new file with rotated_bboxes
            with open(os.path.join(labels_path, file.replace('.jpg', '_rotated.txt')), 'w') as f:
                for bbox in rotated_bboxes:
                    class_name, cx, cy, bbox_width, bbox_height = bbox
                    f.write(f'{int(class_name)} {float(cx)} {float(cy)} {float(bbox_width)} {float(bbox_height)}\n')

            # Now you can save the rotated image and the new bounding boxes coordinates
            cv2.imwrite(os.path.join(images_path, file.replace('.jpg', '_rotated.jpg')), rotated_image)

            # Rotate 60 degrees
            rotated_image, rotated_bboxes = rotate_image_and_bboxes(image, bboxes, 60)

            # Verify if the rotated_bbox is inside the image, if not, ignore this bbox
            rotated_bboxes = verify_bboxes(rotated_bboxes)

            # Create new file with rotated_bboxes
            with open(os.path.join(labels_path, file.replace('.jpg', '_rotated2.txt')), 'w') as f:
                for bbox in rotated_bboxes:
                    class_name, cx, cy, bbox_width, bbox_height = bbox
                    f.write(f'{int(class_name)} {float(cx)} {float(cy)} {float(bbox_width)} {float(bbox_height)}\n')

            # Now you can save the rotated image and the new bounding boxes coordinates
            cv2.imwrite(os.path.join(images_path, file.replace('.jpg', '_rotated2.jpg')), rotated_image)
    
    print(f"Data augmentation done ✅")



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reporoot', type=str, default=ROOT, help='path to repo root')
    opt = parser.parse_args()
    return opt


## Models comparison functions ##

def mean_df(df: pd.DataFrame):
    # Specify the columns for which you want to calculate the mean
    columns_to_mean = ['Precision', 'Recall', 'mAP0-50', 'mAP50-95']

    # Calculate the average values for each model precision, recall, and mAP
    means = []
    for i in range(0, len(df), 2):
        avg = df.iloc[i:i+2][columns_to_mean].mean()
        means.append(avg)

    # Create the new DataFrame with the average values
    new_data = {
        'Model': ['YOLOv5n', 'YOLOv5s', 'YOLOv8n', 'YOLOv8s'],
        'Model Size (MB)': [df.iloc[0]['Model Size (MB)'], df.iloc[2]['Model Size (MB)'], df.iloc[4]['Model Size (MB)'], df.iloc[6]['Model Size (MB)']],
        'Parameters': [df.iloc[0]['Parameters'], df.iloc[2]['Parameters'], df.iloc[4]['Parameters'], df.iloc[6]['Parameters']],
        'Precision': [mean['Precision'] for mean in means],
        'Recall': [mean['Recall'] for mean in means],
        'mAP0-50': [mean['mAP0-50'] for mean in means],
        'mAP50-95': [mean['mAP50-95'] for mean in means]
    }

    return pd.DataFrame(new_data)


def plot_model_size(df: pd.DataFrame):
    sorted_df = df.sort_values('Model Size (MB)')  # Sort DataFrame by 'Model Size (MB)'

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'orange']
    bars = plt.bar(range(len(sorted_df)), sorted_df['Model Size (MB)'], color=colors)

    plt.xlabel('Trained Model')
    plt.ylabel('Model Size (MB)')
    plt.title('Size Comparison')

    plt.xticks(range(len(sorted_df)), sorted_df['Model'])

    for bar, model_name in zip(bars, sorted_df['Model']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
        bar.set_label(model_name)

    plt.legend(loc='upper left')

    # Create a temporary directory to store the plot as an image file
    if not os.path.exists(f'{ROOT}/models/plots'):
        os.makedirs(f'{ROOT}/models/plots')

    plt.savefig(f'{ROOT}/models/plots/01_model_size.jpg')  # Save the plot as an image file

    plt.show()
    
# Parameters and GFLOPs Comparison
def plot_model_params(df: pd.DataFrame):
    sorted_df_params = df.sort_values('Parameters')  # Sort DataFrame by 'Parameters'
    plt.figure(figsize=(10, 6))

    # Plotting Model Parameters
    colors = ['blue', 'green', 'red', 'orange']
    bars_params = plt.bar(range(len(sorted_df_params)), sorted_df_params['Parameters'], color=colors)

    plt.xlabel('Trained Model')
    plt.ylabel('Parameters')
    plt.title('Parameters Comparison')
    plt.xticks(range(len(sorted_df_params)), sorted_df_params['Model'])

    for bar, model_name in zip(bars_params, sorted_df_params['Model']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')
        bar.set_label(model_name)

    # Format y-axis labels
    plt.ticklabel_format(style='plain', axis='y')

    plt.legend(loc='upper left')

    # Create a temporary directory to store the plot as an image file
    if not os.path.exists(f'{ROOT}/models/plots'):
        os.makedirs(f'{ROOT}/models/plots')

    plt.savefig(f'{ROOT}/models/plots/02_model_params.jpg')  # Save the plot as an image file


    plt.show()

    

# Plot the Precision and Recall
def plot_precision_recall(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(df))

    plt.bar(index, df['Precision'], width=bar_width, label='Precision')
    plt.bar(index + bar_width, df['Recall'], width=bar_width, label='Recall')

    plt.xlabel('Trained Model')
    plt.ylabel('Score')
    plt.title('Precision and Recall Comparison')
    plt.xticks(index + bar_width / 2, df['Model'])
    plt.ylim([0.80, 1])
    plt.legend()

    # Create a temporary directory to store the plot as an image file
    if not os.path.exists(f'{ROOT}/models/plots'):
        os.makedirs(f'{ROOT}/models/plots')

    plt.savefig(f'{ROOT}/models/plots/03_model_precision_recall.jpg')  # Save the plot as an image file

    plt.show()

# Plot the mAP50-95
def plot_mAP(df: pd.DataFrame):
    sorted_df = df.sort_values('mAP50-95')  # Sort DataFrame by 'mAP50-95'

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'orange']
    bars = plt.bar(range(len(sorted_df)), sorted_df['mAP50-95'], color=colors)

    plt.xlabel('Trained Model')
    plt.ylabel('mAP50-95')
    plt.title('mAP 50-95% Comparison')

    plt.xticks(range(len(sorted_df)), sorted_df['Model'])

    for bar, model_name in zip(bars, sorted_df['Model']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.3f}', ha='center', va='bottom')
        bar.set_label(model_name)

    plt.legend(loc='upper left')

    plt.ylim([0.6, 1])

    # Create a temporary directory to store the plot as an image file
    if not os.path.exists(f'{ROOT}/models/plots'):
        os.makedirs(f'{ROOT}/models/plots')

    plt.savefig(f'{ROOT}/models/plots/04_model_map.jpg')  # Save the plot as an image file

    plt.show()

# Get images from plots
def save_plots(path: str = f'{ROOT}/models/plots/'):
    # Let's combine the plots into one image, vertically
    images = []
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            images.append(Image.open(os.path.join(path, file)))

    widths, heights = zip(*(i.size for i in images))

    max_width = max(widths)
    total_height = sum(heights)

    new_im = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for im in images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.size[1]
   
    # Save the combined image
    new_im.save(f'{ROOT}/models/models_comparison.jpg')


import pandas as pd

def get_results_df(df: pd.DataFrame, yolov5n_df: pd.DataFrame, yolov5s_df: pd.DataFrame, yolov8n_df: pd.DataFrame, yolov8s_df: pd.DataFrame):
    results = {
        'Model': ['YOLOv5n', 'YOLOv5s', 'YOLOv8n', 'YOLOv8s'],
        'Cars Accuracy': [
            (yolov5n_df['Cars'] == df['Cars']).mean(),
            (yolov5s_df['Cars'] == df['Cars']).mean(),
            (yolov8n_df['Cars'] == df['Cars']).mean(),
            (yolov8s_df['Cars'] == df['Cars']).mean()
        ],
        'Occupied disabled parking spots Accuracy': [
            (yolov5n_df['Occupied disabled parking spots'] == df['Occupied disabled parking spots']).mean(),
            (yolov5s_df['Occupied disabled parking spots'] == df['Occupied disabled parking spots']).mean(),
            (yolov8n_df['Occupied disabled parking spots'] == df['Occupied disabled parking spots']).mean(),
            (yolov8s_df['Occupied disabled parking spots'] == df['Occupied disabled parking spots']).mean()
        ],
        'Empty disabled parking spots Accuracy': [
            (yolov5n_df['Empty disabled parking spots'] == df['Empty disabled parking spots']).mean(),
            (yolov5s_df['Empty disabled parking spots'] == df['Empty disabled parking spots']).mean(),
            (yolov8n_df['Empty disabled parking spots'] == df['Empty disabled parking spots']).mean(),
            (yolov8s_df['Empty disabled parking spots'] == df['Empty disabled parking spots']).mean()
        ],
        'Occupied parking spots Accuracy': [
            (yolov5n_df['Occupied parking spots'] == df['Occupied parking spots']).mean(),
            (yolov5s_df['Occupied parking spots'] == df['Occupied parking spots']).mean(),
            (yolov8n_df['Occupied parking spots'] == df['Occupied parking spots']).mean(),
            (yolov8s_df['Occupied parking spots'] == df['Occupied parking spots']).mean()
        ],
        'Empty parking spots Accuracy': [
            (yolov5n_df['Empty parking spots'] == df['Empty parking spots']).mean(),
            (yolov5s_df['Empty parking spots'] == df['Empty parking spots']).mean(),
            (yolov8n_df['Empty parking spots'] == df['Empty parking spots']).mean(),
            (yolov8s_df['Empty parking spots'] == df['Empty parking spots']).mean()
        ],
        'Cars in transit or parked in non-parking spots Accuracy': [
            (yolov5n_df['Cars in transit or parked in non-parking spots'] == df['Cars in transit or parked in non-parking spots']).mean(),
            (yolov5s_df['Cars in transit or parked in non-parking spots'] == df['Cars in transit or parked in non-parking spots']).mean(),
            (yolov8n_df['Cars in transit or parked in non-parking spots'] == df['Cars in transit or parked in non-parking spots']).mean(),
            (yolov8s_df['Cars in transit or parked in non-parking spots'] == df['Cars in transit or parked in non-parking spots']).mean()
        ],
    }

    return pd.DataFrame.from_dict(results)
