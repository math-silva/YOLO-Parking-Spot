import os
import shutil
import argparse
import yaml

from functions import split_dataset
from yolov5start import parse_opt, only_car_label

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

def update_yaml(yaml_path, yolo_path):
    with open(yaml_path, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
        yaml_file['path'] = os.path.join(yolo_path, 'data/')
        yaml_file['train'] = os.path.join(yolo_path, 'data/train.txt')
        yaml_file['val'] = os.path.join(yolo_path, 'data/val.txt')

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_file, f)


def copy_data2yolov8(repo_path, yolo_path):
    # get data dir path
    data_path = os.path.join(repo_path, 'data')

    # now copy data dir to yolov5/data/custom-data
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/'))
    print(f"Data copied to {os.path.join(yolo_path, 'data/')} ✅")

    # Now, let's split the dataset
    split_dataset(os.path.join(yolo_path, 'data/'))

    # now change yolov8/data/dataset.yaml path to yolov8/dataset.yaml
    shutil.copyfile(os.path.join(yolo_path, 'data/dataset.yaml'), os.path.join(yolo_path, 'dataset.yaml'))
    
    # now delete yolov8/data/dataset.yaml
    os.remove(os.path.join(yolo_path, 'data/dataset.yaml'))
    update_yaml(os.path.join(yolo_path, 'dataset.yaml'), yolo_path)


def main(opt):
    # get current working directory
    repo_path = opt.reporoot
    
    yolo_path = os.path.join(repo_path, 'yolov8')

    # if yolo path exists, delete it
    if os.path.exists(yolo_path):
        shutil.rmtree(yolo_path)

    print(f"Repo path: {repo_path}")
    print(f"Yolo path: {yolo_path}")

    copy_data2yolov8(repo_path, yolo_path)
    only_car_label(os.path.join(yolo_path, 'data/labels'))
    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
