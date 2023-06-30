import os
import shutil
import yaml

from functions import split_dataset, data_augmentation, only_car_label, parse_opt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def update_yaml(yaml_path, fold):
    with open(yaml_path, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
        yaml_file['path'] = 'data/custom-data/' + fold

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_file, f)


def copy_data2yolov5(repo_path, yolo_path):
    # get data dir path
    data_path = os.path.join(repo_path, 'data')

    # if yolov5/data/custom-data/exists, delete it
    if os.path.exists(os.path.join(yolo_path, 'data/custom-data')):
        shutil.rmtree(os.path.join(yolo_path, 'data/custom-data'))

    # now copy data dir to yolov5/data/custom-data/fold_0 and yolov5/data/custom-data/fold_1
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/custom-data/fold_0'))
    shutil.copytree(data_path, os.path.join(yolo_path, 'data/custom-data/fold_1'))
    print(f"Data copied to {os.path.join(yolo_path, 'data/custom-data')} ✅")

    update_yaml(os.path.join(yolo_path, 'data/custom-data/fold_0/dataset.yaml'), 'fold_0')
    update_yaml(os.path.join(yolo_path, 'data/custom-data/fold_1/dataset.yaml'), 'fold_1')


def main(opt):
    # Get current working directory
    repo_path = opt.reporoot
    # Get yolov5 path
    yolo_path = os.path.join(repo_path, 'yolov5')

    print(f"Repo path: {repo_path}")
    print(f"Yolo path: {yolo_path}")

    # Copy data to yolov5/data/custom-data
    copy_data2yolov5(repo_path, yolo_path)

    fold_0_path = os.path.join(yolo_path, 'data/custom-data/fold_0')
    fold_1_path = os.path.join(yolo_path, 'data/custom-data/fold_1')

    # Delete all labels that are not cars
    only_car_label(os.path.join(fold_0_path, 'labels'))
    only_car_label(os.path.join(fold_1_path, 'labels'))

    # Now, let's split the dataset
    split_dataset(os.path.join(yolo_path, 'data/custom-data'), 0.5)

    # Data augmentation
    data_augmentation(fold_0_path, os.path.join(fold_0_path, 'train.txt'))
    data_augmentation(fold_1_path, os.path.join(fold_1_path, 'train.txt'))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
    
