# <div align="center">YOLO Parking Spot</div>

## Overview
In this project, we trained four YOLO models on images get by drone from the Unifesp (Federal University of São Paulo) parking lot to detect cars. The goal is to gather information and potentially develop a parking solution to enhance traffic flow and optimize space utilization. By identifying cars, we generate a report with image details such as the number of available parking spots, the number of cars in transit, parked cars, etc.

We documented our project in a research paper, providing detailed explanations of our methodologies, experimental setup, and results. Additionally, we created a YouTube video where we showcased the project and demonstrated its functionalities.
<div align="center">
  <img width="80%" src="images/project-banner.png">
</div>

## Table of contents
- [Authors](#authors)
- [YOLO Introduction](#yolo-introduction)
- [Training](#training)
- [Model Usage](#model-usage)
- [Model Comparison](#model-comparison)
  - [Size](#size)
  - [Parameters](#parameters)
  - [Precision and Recall](#precision-and-recall)
  - [mAP](#map-50-95)
- [Results Comparison](#results-comparison)
- [Conclusion](#conclusion)
- [Additional Information](#additional-information)
    - [YouTube Video](#youtube-video)
    - [Paper](#paper)
- [License](#license)

## Authors
- [Matheus Silva](https://www.github.com/coding-math) | [LinkedIn](https://www.linkedin.com/in/matheussmsilva/)
- [Marcos Lucas](https://www.github.com/lmarcosz) | [LinkedIn](https://www.linkedin.com/in/marcos-l-silva/)
  
## YOLO Introduction
YOLO (You Only Look Once) is an object detection algorithm that performs real-time object detection in images. It's widely used due to its speed and accuracy. In our project, we utilized YOLOv5 and YOLOv8 models for training. You can find the repository for YOLOv5 [here](https://github.com/ultralytics/yolov5) and YOLOv8 [here](https://github.com/ultralytics/ultralytics).

## Training
To train our models, we employed specific strategies. Since we only had 23 images, we decided to use the 2-fold cross-validation method. This approach involves dividing the dataset into two subsets: training and testing.

To address the limited amount of training data, we applied a data augmentation algorithm to the training images. In addition to tripling the number of training images by performing rotations of 30º and 60º, this augmentation technique allowed our models to learn to identify cars with better accuracy. By introducing variations in the training dataset, we enhanced the model's ability to generalize and detect cars from different perspectives.
<div align="center">
  <img width="70%" src="images/data_augmentation_horizontal.png">
</div>

During the training process, the models were trained using the YOLO (You Only Look Once) algorithm. This algorithm is widely used for object detection tasks. The training involves optimizing the model's parameters to accurately detect cars in the parking lot images.

One important aspect of the training process is the IoU (Intersection over Union) threshold. The IoU threshold determines the level of overlap required between the predicted bounding box and the ground truth bounding box to consider it a correct detection. Typically, a threshold of 0.5 is used, which means that the predicted bounding box must have an overlap of at least 50% with the ground truth bounding box to be considered a valid detection.
<div align="center">
  <img width="70%" src="images/iou-example.png">
</div>

In the example image, we can observe the evaluation of two bounding boxes. On the left side, there is a false positive example with an IoU threshold of 0.3. The predicted bounding box, shown in red, has a low overlap with the ground truth bounding box, depicted in green. Since the IoU falls below the threshold, this detection is considered incorrect.

On the right side, we have a true positive example with an IoU of 0.8. The predicted bounding box, again shown in red, closely aligns with the ground truth bounding box, represented in green. With an IoU above the specified threshold, this detection is considered accurate.

The IoU threshold plays a crucial role in determining the quality of the detections during training, allowing for a balance between false positives and false negatives.

## Model Usage
Once our trained model receives an image, it predicts the positions of all the cars in that image. We then utilize an algorithm to extract information from predefined parking spaces. By analyzing the car positions provided by the model, we accurately determine the number of available parking spots, occupied spots, and cars in transit or non-parking areas.

To understand the process better, let's look at the step-by-step breakdown:

1. **Input Image**: The initial image captured by the drone.
<div align="center">
  <img width="70%" src="data/images/occupied_set1_40m_1.jpg">
</div>
<br>

2. **Prediction Image**: The image generated by our model, highlighting the positions of cars predicted by the model.
<div align="center">
  <img width="70%" src="results/yolov5n_fold_0/yolo_images/occupied_set1_40m_1.jpg">
</div>
<br>

3. **Algorithm Image**: The final image generated by our algorithm, showing the extracted information about parking spots, such as available spots, occupied spots, and cars in transit or non-parking areas.
<div align="center">
  <img width="70%" src="results/yolov5n_fold_0/images/occupied_set1_40m_1.jpg">
</div>

## Model Comparison

We trained a total of eight models, utilizing the 2-fold strategy for each YOLO model: YOLOv5n, YOLOv5s, YOLOv8n, and YOLOv8s. This means we have four models for comparison, obtained by averaging the results from the trained models.

### Size
![Size Comparison Graphic](models/plots/01_model_size.jpg)

The size analysis reveals notable differences among the YOLO models. The YOLOv5n and YOLOv8n models exhibit smaller file sizes, at 4.8MB and 6.5MB, respectively. On the other hand, the YOLOv5s and YOLOv8s models have larger sizes, measuring 15.3MB and 22.7MB, respectively.

### Parameters
![Parameters Comparison Graphic](models/plots/02_model_params.jpg)

Analyzing the parameters, we observe varying complexities across the YOLO models. The YOLOv5n and YOLOv8n models have lower parameter counts, with 1,760,518 and 3,005,843 parameters, respectively. In contrast, the YOLOv5s and YOLOv8s models exhibit higher parameter counts, totaling 7,012,822 and 11,125,971 parameters, respectively.

### Precision and Recall
![Precision and Recall Comparison Graphic](models/plots/03_model_precision_recall.jpg)

The precision and recall analysis sheds light on the performance of the YOLO models in terms of object detection accuracy. Comparing the models, we can observe that YOLOv5s demonstrates the highest precision and recall, with values of 0.998 and 0.999 respectively. YOLOv5n follows closely with precision and recall values of 0.992 and 0.996 respectively. Meanwhile, YOLOv8n and YOLOv8s exhibit slightly lower precision and recall scores, with YOLOv8n achieving a precision of 0.983 and recall of 0.980, and YOLOv8s attaining a precision of 0.976 and recall of 0.969. These results provide insights into the models' ability to accurately detect objects, with YOLOv5s standing out as the top performer in terms of precision and recall, closely followed by YOLOv5n.

### mAP 50-95
![mAP Comparison Graphic](models/plots/04_model_map.jpg)

The mean Average Precision (mAP) in the 50-95 threshold range provides an overall assessment of the models' object detection performance. Comparing the mAP scores, we can see that YOLOv8n achieved the highest mAP of 0.755, indicating its ability to consistently detect objects across a wide range of IoU thresholds. YOLOv5s follows closely with an mAP of 0.737, showcasing its strong performance in detecting objects with high precision and recall. YOLOv5n demonstrates a respectable mAP of 0.712, while YOLOv8s achieved an mAP of 0.719. These mAP scores highlight the models' effectiveness in accurately localizing and identifying objects, with YOLOv8n exhibiting the highest performance among the compared models in the 50-95 threshold range.

## Results Comparison

To assess the accuracy of our models, we calculated various metrics, including accuracy of the car count, occupied parking spots, empty parking spots, and cars in transit.

| Model   | Cars Accuracy | Occupied disabled parking spots Accuracy | Empty disabled parking spots Accuracy | Occupied parking spots Accuracy | Empty parking spots Accuracy | Cars in transit or parked in non-parking spots Accuracy |
|---------|----------|----------|----------|----------|---------|----------|
| YOLOv5n | 0.695    | 1.000    | 1.000    | 1.000    | 1.000   | 0.695    |
| YOLOv5s | 0.695    | 1.000    | 1.000    | 1.000    | 1.000   | 0.695    |
| YOLOv8n | 0.739    | 1.000    | 1.000    | 1.000    | 1.000   | 0.739    |
| YOLOv8s | 0.695    | 1.000    | 1.000    | 0.739    | 0.739   | 0.782    |

The table above presents the Accuracy values for different columns based on predictions made by various models. These values indicate the models' performance in their respective tasks. Overall, the models achieved good results, particularly in identifying occupied and empty parking spots.

However, an interesting observation arises concerning the "Cars" column. The Accuracy values for this column reveal that the models made some errors in predicting the number of cars identified in the images. Further analysis revealed that these errors were not due to shortcomings of the models themselves. Instead, the errors occurred because there were instances where cars were not manually marked during the dataset creation process.

In other words, the models were able to detect cars in certain positions that were initially overlooked during the manual annotation of the dataset. This realization highlights the models' proficiency in identifying cars, even in unexpected positions. The errors observed in the "Cars" column can be attributed to human error during the dataset labeling phase, where some cars were incorrectly assumed to be undetectable by the models.

These results emphasize the models' capabilities in accurately identifying occupied and empty parking spots. They also underscore the importance of thorough and precise dataset labeling to ensure accurate evaluation and performance assessment of the models.

## Conclusion

Based on the analysis presented, it can be concluded that all four models achieved excellent results for the parking lot at ICT Unifesp. However, based on certain factors, the YOLOv5n model can be considered the best option for this particular case.

The YOLOv5n model demonstrated impressive performance in accurately identifying occupied and empty parking spots, as indicated by the consistently great Accuracy values across multiple columns. Additionally, it has the advantage of being the lightest in terms of file size (in MB) and having the fewest parameters compared to the other models. This translates into faster processing speed and lower memory usage, making it a practical choice for real-time applications.

To further improve our work, it is recommended to create a larger and more precisely labeled dataset. By expanding the dataset and ensuring accurate labeling of cars and corresponding labels, we can enhance the training process and increase the model's ability to detect cars in various positions and conditions accurately.

By investing in a more comprehensive and accurately labeled dataset, we can train a model that is tailored specifically to the ICT Unifesp parking lot, optimizing its performance and ensuring it meets the specific requirements of our study. With continued efforts to refine the dataset and fine-tune the model, we can strive for even better results and contribute to the advancement of parking spot detection and monitoring systems.

## Additional Information
#### Note: The paper and video are in Brazilian Portuguese.

### <div id="youtube-video" align="center">[YouTube Video](https://www.youtube.com/watch?v=XJYMB878Iag)</div>
### <div align="center">Identificação de Veículos em Estacionamentos: Uma Abordagem Comparativa Entre Modelos YOLO</div>

<div align="center">
    <a href="https://www.youtube.com/watch?v=XJYMB878Iag">
        <img width="30%" src="images/thumbnail.png">
    </a>
</div>
<br>

### <div id="paper" align="center">[Paper](paper.pdf)</div>
### <div align="center">Identificação de Veículos em Estacionamentos: Uma Abordagem Comparativa Entre Modelos YOLO</div>

<div align="center">
    <a href="paper.pdf">
        <img width="30%" src="images/paper-preview.png">
    </a>
</div>

## License
The **YOLO Parking Spot** project is licensed under the MIT License. Feel free to use and modify the code according to your needs.
