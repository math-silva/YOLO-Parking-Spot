# utils/functions.py

# pylint: disable=unsubscriptable-object
# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none

import numpy as np
import cv2
import os

# Function to check if a car is occupying a parking spot
def is_occupied(image, annotations, left, top, right, bottom, threshold):
    # Get the car bounding boxes
    car_boxes = []
    for annotation in annotations:
        class_id, x_center, y_center, width, height = map(float, annotation.split())
        if class_id == 4:
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
    disabled_spot_occupied = 0
    spot_occupied = 0
    
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
        if class_id == 4: # Car
            if highlighted_cars:
                color = (0, 165, 255)  # Orange color
            else:
                continue
        elif class_id == 15: # Disabled parking spot
            # Check if there is a car occupying of the parking spot
            if is_occupied(image, annotations, left, top, right, bottom, threshold):
                disabled_spot_occupied += 1
                color = (0, 0, 255)  # Red color
            else:
                color = (255, 0, 0)  # Blue color
        elif class_id == 16: # Parking spot
            # Check if there is a car occupying the parking spot
            if is_occupied(image, annotations, left, top, right, bottom, threshold):
                spot_occupied += 1
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
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)

    alpha = 0.4  # Transparency factor.
    
    # Image legend
    text_position = (image.shape[1] - 400, 30)  # Top-right corner position
    overlay = image.copy()
    cv2.rectangle(overlay, (text_position[0] - 10, text_position[1] - 30), (text_position[0] + 390, text_position[1] + 80), (0, 0, 0), cv2.FILLED)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) # Add the overlay to the image
    
    text = f'Disabled parking spots Count: {class_counts.get(15, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate
    
    text = f'Parking spots Count: {class_counts.get(16, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Cars count: {class_counts.get(4, 0)}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    text_position = (30, 30)  # Top-left corner position
    overlay = image.copy()
    cv2.rectangle(overlay, (text_position[0] - 10, text_position[1] - 30), (text_position[0] + 575, text_position[1] + 140), (0, 0, 0, 0.8), cv2.FILLED)
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0) # Add the overlay to the image

    text = f'Empty disabled parking spots Count: {class_counts.get(15, 0) - disabled_spot_occupied}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Occupied disabled parking spots Count: {disabled_spot_occupied}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Empty parking spots Count: {class_counts.get(16, 0) - spot_occupied}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Occupied parking spots count: {spot_occupied}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    text_position = (text_position[0], text_position[1] + 30)  # Increment the y-coordinate

    text = f'Cars in transit or parked in non-parking spots: {class_counts.get(4, 0) - disabled_spot_occupied - spot_occupied}'
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Save the image with bounding boxes
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)


# Function to process all images in a folder
def process_images(folder_path, output_folder, threshold=0.4, highlighted_cars=True):
    processed_images = 0
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image and annotation in the folder
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.jpg'):  # Assuming image files have the .jpg extension
            # Get the paths for the current image and its corresponding annotation file
            image_path = os.path.join(folder_path, image_file)
            annotation_file = os.path.splitext(image_file)[0] + '.txt'
            annotation_path = os.path.join(folder_path, annotation_file)
            
            # Call the function to draw bounding boxes and save the resulting image
            draw_bounding_boxes(image_path, annotation_path, output_folder, threshold, highlighted_cars)
            processed_images += 1

    print(f'Processed {processed_images} images')