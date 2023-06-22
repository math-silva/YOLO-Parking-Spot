import os
import shutil
import argparse

from functions import split_dataset

# root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def copy_data2yolov5(repo_path, yolo_path):
    # get data dir path
    data_path = os.path.join(repo_path, 'data')

    # if yolov5/data/custom-data exists, delete it
    if os.path.exists(os.path.join(yolo_path, 'data/custom-data')):
        shutil.rmtree(os.path.join(yolo_path, 'data/custom-data'))

    # now copy data dir to yolov5/data/custom-data
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/custom-data'))
    print(f"Data copied to {os.path.join(yolo_path, 'data/custom-data')} ✅")

    # Now, let's split the dataset
    split_dataset(os.path.join(yolo_path, 'data/custom-data/'))


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


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reporoot', type=str, default=ROOT, help='path to repo root')
    opt = parser.parse_args()
    return opt


def main(opt):
    # get current working directory
    repo_path = opt.reporoot
    # get yolov5 path
    yolo_path = os.path.join(repo_path, 'yolov5')

    print(f"Repo path: {repo_path}")
    print(f"Yolo path: {yolo_path}")

    copy_data2yolov5(repo_path, yolo_path)
    only_car_label(os.path.join(yolo_path, 'data/custom-data/labels'))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    
