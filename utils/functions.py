# utils/functions.py

# pylint: disable=unsubscriptable-object
# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none

import numpy as np
import pandas as pd
import cv2
import os
import shutil

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
    text_position = (image.shape[1] - 400, 30)  # Top-right corner position
    overlay = image.copy()
    cv2.rectangle(overlay, (text_position[0] - 10, text_position[1] - 30), (text_position[0] + 390, text_position[1] + 80), (0, 0, 0), cv2.FILLED) # type: ignore
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) # Add the overlay to the image
    
    text = f'Disabled parking spots count: {class_counts.get(1, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate
    
    text = f'Parking spots count: {class_counts.get(2, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Cars count: {class_counts.get(0, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    text_position = (30, 30)  # Top-left corner position
    overlay = image.copy()
    cv2.rectangle(overlay, (text_position[0] - 10, text_position[1] - 30), (text_position[0] + 575, text_position[1] + 140), (0, 0, 0, 0.8), cv2.FILLED) # type: ignore
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) # Add the overlay to the image

    text = f'Empty disabled parking spots count: {empty_disabled_spot}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Occupied disabled parking spots count: {occupied_disabled_spot}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Empty parking spots count: {empty_spot}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Occupied parking spots count: {occupied_spot}'
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
        'Disabled parking spots count': [class_counts.get(1, 0)],
        'Parking spots count': [class_counts.get(2, 0)],
        'Cars count': [class_counts.get(0, 0)],
        'Empty disabled parking spots count': [empty_disabled_spot],
        'Occupied disabled parking spots count': [occupied_disabled_spot],
        'Empty parking spots count': [empty_spot],
        'Occupied parking spots count': [occupied_spot],
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
        if model.__contains__('/'): # if model is a path
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

    # Save the results in a dataframe
    columns = ['Image File', 'Disabled parking spots count', 'Parking spots count', 'Cars count',
            'Empty disabled parking spots count', 'Occupied disabled parking spots count',
            'Empty parking spots count', 'Occupied parking spots count',
            'Cars in transit or parked in non-parking spots']

    # Create an empty dataframe
    results_df = pd.DataFrame(columns=columns)

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