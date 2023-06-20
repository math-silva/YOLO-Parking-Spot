import os
import shutil


def copy_data2yolo(repo_path, yolo_path):
    # get data dir path
    data_path = os.path.join(repo_path, 'data')

    # now copy data dir to yolov5/data/custom-data
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/custom-data'))
    print(f"Data copied to {os.path.join(yolo_path, 'data/custom-data')} ✅")


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


# get current working directory
repo_path = os.getcwd()
yolo_path = os.path.join(repo_path, 'yolov5')

copy_data2yolo(repo_path, yolo_path)
only_car_label(os.path.join(yolo_path, 'data/custom-data/labels'))




