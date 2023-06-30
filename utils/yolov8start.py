import os
import shutil
import yaml

from functions import split_dataset, data_augmentation, only_car_label, parse_opt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

def update_yaml(yaml_path, fold_path):
    with open(yaml_path, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
        yaml_file['path'] = fold_path
        yaml_file['train'] = os.path.join(fold_path, 'train.txt')
        yaml_file['val'] = os.path.join(fold_path, 'val.txt')

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_file, f)


def copy_data2yolov8(repo_path, yolo_path):
    # Get data dir path
    data_path = os.path.join(repo_path, 'data')

    fold_0_path = os.path.join(yolo_path, 'data/fold_0')
    fold_1_path = os.path.join(yolo_path, 'data/fold_1')
    
    # Now copy data dir to yolov8/data/fold_x
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/fold_0'))
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/fold_1'))
    print(f"Data copied to {os.path.join(yolo_path, 'data/')} ✅")

    # now change yolov8/data/fold_x/dataset.yaml path to yolov8/dataset_fold_0.yaml, yolov8/dataset_fold_1.yaml
    shutil.copyfile(os.path.join(yolo_path, 'data/fold_0/dataset.yaml'), os.path.join(yolo_path, 'dataset_fold_0.yaml'))
    shutil.copyfile(os.path.join(yolo_path, 'data/fold_1/dataset.yaml'), os.path.join(yolo_path, 'dataset_fold_1.yaml'))
    
    # now delete yolov8/data/fold_x/dataset.yaml
    os.remove(os.path.join(yolo_path, 'data/fold_0/dataset.yaml'))
    os.remove(os.path.join(yolo_path, 'data/fold_1/dataset.yaml'))

    update_yaml(os.path.join(yolo_path, 'dataset_fold_0.yaml'), os.path.join(yolo_path, 'data/fold_0')) # fold_0
    update_yaml(os.path.join(yolo_path, 'dataset_fold_1.yaml'), os.path.join(yolo_path, 'data/fold_1')) # fold_1

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

    fold_0_path = os.path.join(yolo_path, 'data/fold_0')
    fold_1_path = os.path.join(yolo_path, 'data/fold_1')

    # Delete all labels that are not cars
    only_car_label(os.path.join(fold_0_path, 'labels'))
    only_car_label(os.path.join(fold_1_path, 'labels'))

    # Now, let's split the dataset
    split_dataset(os.path.join(yolo_path, 'data/'), 0.5)
    
    # Data augmentation
    data_augmentation(fold_0_path, os.path.join(fold_0_path, 'train.txt'))
    data_augmentation(fold_1_path, os.path.join(fold_1_path, 'train.txt'))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
