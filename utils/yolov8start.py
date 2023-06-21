import os
import shutil

from yolov5start import parse_opt, copy_data2yolo, only_car_label

def main(opt):
    # get current working directory
    repo_path = opt.reporoot
    
    yolo_path = os.path.join(repo_path, 'yolov8')

    print(f"Repo path: {repo_path}")

    copy_data2yolo(repo_path, yolo_path)
    only_car_label(os.path.join(yolo_path, 'data/custom-data/labels'))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
