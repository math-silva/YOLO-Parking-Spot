import os
import shutil
import yaml

from functions import split_dataset, data_augmentation, only_car_label, parse_opt

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
    # Get data dir path
    data_path = os.path.join(repo_path, 'data')

    # Now copy data dir to yolov5/data/custom-data
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/'))
    print(f"Data copied to {os.path.join(yolo_path, 'data/')} ✅")

    # now change yolov8/data/dataset.yaml path to yolov8/dataset.yaml
    shutil.copyfile(os.path.join(yolo_path, 'data/dataset.yaml'), os.path.join(yolo_path, 'dataset.yaml'))
    
    # now delete yolov8/data/dataset.yaml
    os.remove(os.path.join(yolo_path, 'data/dataset.yaml'))
    update_yaml(os.path.join(yolo_path, 'dataset.yaml'), yolo_path)


def main(opt):
    # Get current working directory
    repo_path = opt.reporoot
    
    # Get yolov8 path
    yolo_path = os.path.join(repo_path, 'yolov8')

    # If yolo path exists, delete it
    if os.path.exists(yolo_path):
        shutil.rmtree(yolo_path)

    print(f"Repo path: {repo_path}")
    print(f"Yolo path: {yolo_path}")

    # Copy data to yolov8/data/
    copy_data2yolov8(repo_path, yolo_path)

    # Delete all labels that are not cars
    only_car_label(os.path.join(yolo_path, 'data/labels'))

    # Data augmentation
    data_augmentation(os.path.join(yolo_path, 'data/'))
    
    # Now, let's split the dataset
    split_dataset(os.path.join(yolo_path, 'data/'))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
