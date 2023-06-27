import os
import shutil

from functions import split_dataset, data_augmentation, only_car_label, parse_opt

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


def main(opt):
    # Get current working directory
    repo_path = opt.reporoot
    # Get yolov5 path
    yolo_path = os.path.join(repo_path, 'yolov5')

    print(f"Repo path: {repo_path}")
    print(f"Yolo path: {yolo_path}")

    # Copy data to yolov5/data/custom-data
    copy_data2yolov5(repo_path, yolo_path)

    # Delete all labels that are not cars
    only_car_label(os.path.join(yolo_path, 'data/custom-data/labels'))

    # Data augmentation
    data_augmentation(os.path.join(yolo_path, 'data/custom-data/'))
    
    # Now, let's split the dataset
    split_dataset(os.path.join(yolo_path, 'data/custom-data/'))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    
