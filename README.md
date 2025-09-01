# A-Comparative-Analysis-of-Object-Detection-Models-for-Plant-Leaf-Counting-under-Overlapping-Foliage-Conditions
A Comparative Analysis of three foundational object detection models: YOLOv8, YOLOv12, and Faster R-CNN for leaf cointing in dense foliage. Evaluated on all performance metrics: mAP50, mAP50-95, IOU, MAE, and MAPE.

## Overview
This project compares three computer vision models: YOLOv8, YOLOv12, and Faster R-CNN. The script provided is for training and evaluating the three models on multiple datasets.

### Model Performance Comparison
| Model      | Size | Epochs | Precision | Recall | mAP@.50 | mAP@.50-.95 | Batch | Image Size | Dataset         | IoU  | MSE (Count) | MAE (Count) | MAPE   |
|------------|------|--------|-----------|--------|---------|-------------|-------|------------|-----------------|------|-------------|-------------|--------|
| YOLOv8     | N    | 50     | 0.657     | 0.483  | 0.546   | 0.237       | 16    | 640x640    | Default         | 0.584| 4684.22     | 40.65       | 31.80% |
| YOLOv8     | M    | 50     | 0.738     | 0.553  | 0.625   | 0.287       | 16    | 640x640    | Default         | 0.602| 2987.15     | 30.82       | 25.06% |
| YOLOv8     | L    | 60     | 0.721     | 0.571  | 0.635   | 0.295       | 16    | 640x640    | Default         | 0.623| 3043.60     | 33.67       | 27.34% |
| YOLOv8     | L    | 100    | 0.745     | 0.569  | 0.636   | 0.294       | 16    | 640x640    | Default         | 0.612| 2193.25     | 27.48       | 24.89% |
| YOLOv8     | M    | 100    | 0.775     | 0.682  | 0.743   | 0.395       | 8     | 960x960    | Default         | 0.668| 812.73      | 16.67       | 15.04% |
| YOLOv8     | M    | 100    | 0.797     | 0.674  | 0.753   | 0.394       | 8     | 960x960    | Default         | 0.671| 1188.72     | 19.25       | 15.21% |
| YOLOv8     | M    | 100    | 0.796     | 0.687  | 0.758   | 0.395       | 8     | 960x960    | Default         | 0.671| 1188.72     | 19.25       | 15.21% |
| YOLOv8     | N    | 100    | 0.747     | 0.587  | 0.677   | 0.334       | 16    | 960x960    | Default         | 0.591| 1379.40     | 22.27       | 22.16% |
| YOLOv8     | N    | 100    | 0.702     | 0.542  | 0.612   | 0.280       | 8     | 960x960    | Default         | 0.581| 1627.95     | 23.72       | 21.50% |
| YOLOv8     | N    | 100    | 0.746     | 0.597  | 0.678   | 0.330       | 16    | 960x960    | Default         | 0.591| 1539.63     | 22.37       | 22.95% |
| YOLOv8     | M    | 100    | 0.788     | 0.652  | 0.717   | 0.378       | 16    | 960x960    | Default         | 0.644| 473.02      | 14.35       | 14.13% |
| YOLOv8     | M    | 100    | 0.780     | 0.649  | 0.707   | 0.351       | 16    | 960x960    | Default         | 0.615| 553.55      | 15.92       | 16.82% |
| YOLOv8     | M    | 100    | 0.792     | 0.739  | 0.797   | 0.446       | 8     | 1216x1216  | Tiled           | 0.632| 2266.30     | 30.50       | 25.27% |
| YOLOv12    | S    | 100    | 0.434     | 0.312  | 0.297   | 0.104       | 16    | 640x640    | Default         | 0.515| 8248.35     | 54.45       | 50.11% |
| YOLOv12    | M    | 100    | 0.452     | 0.349  | 0.335   | 0.119       | 16    | 640x640    | Default         | 0.515| 5465.83     | 44.13       | 41.67% |
| YOLOv12    | L    | 100    | 0.485     | 0.344  | 0.345   | 0.124       | 8     | 640x640    | Default         | 0.525| 5693.27     | 44.30       | 40.78% |
| YOLOv12    | X    | 100    | 0.418     | 0.311  | 0.292   | 0.100       | 8     | 640x640    | Default         | 0.515| 8248.35     | 54.45       | 50.11% |
| YOLOv12    | S    | 100    | 0.519     | 0.339  | 0.370   | 0.148       | 8     | 640x640    | Default         | 0.550| 4591.43     | 41.43       | 33.37% |
| YOLOv12    | M    | 100    | 0.576     | 0.377  | 0.418   | 0.174       | 8     | 640x640    | Default         | 0.543| 2773.67     | 30.07       | 25.14% |
| YOLOv12    | L    | 100    | 0.601     | 0.375  | 0.423   | 0.177       | 8     | 640x640    | Default         | 0.577| 4411.10     | 41.00       | 33.41% |
| YOLOv12    | X    | 100    | 0.613     | 0.395  | 0.450   | 0.191       | 8     | 640x640    | Default         | 0.564| 3136.10     | 34.57       | 28.78% |
| YOLOv12    | S    | 100    | 0.604     | 0.432  | 0.476   | 0.209       | 8     | 960x960    | Default         | 0.563| 2211.85     | 28.05       | 25.21% |
| YOLOv12    | M    | 100    | 0.659     | 0.489  | 0.548   | 0.250       | 4     | 960x960    | Default         | 0.567| 1033.22     | 21.28       | 21.28% |
| YOLOv12    | L    | 100    | 0.681     | 0.484  | 0.552   | 0.257       | 4     | 960x960    | Default         | 0.591| 1429.02     | 22.62       | 20.46% |
| YOLOv12    | X    | 100    | 0.692     | 0.493  | 0.563   | 0.262       | 2     | 960x960    | Default         | 0.604| 1387.13     | 23.27       | 21.42% |
| YOLOv12    | S    | 100    | 0.737     | 0.636  | 0.718   | 0.391       | 2     | 1216x1216  | Tiled           | 0.656| 27.58       | 3.43        | 26.84% |
| Faster R-CNN| N/A | 100    | 0.729     | 0.456   | 0.417  | 0.182       | N/A   | 800x1333px | Defaut          | 0.605| 6128.90     | 44.43       | 34.29% |

### Datasets
The datasets used in this project is hosted on Google Drive. You can access the data to download via the following links:
- [raw_dataset](https://drive.google.com/drive/folders/1SqZVu7KqB2eOgOt82ETEhWRWwm2cNwmN?usp=sharing)
- [train_split_complete_dataset](https://drive.google.com/drive/folders/13MaM3n2fJA5EUR2EBFwm7Byr42MP6Ge3?usp=sharing)
- [tiled_dataset](https://drive.google.com/drive/folders/1tl77CQ1zjtLOq4uHEfYI4_9YmdYOTDJL?usp=sharing)

**Note**: All models used the train_split_complete_dataset for training. Only YOLO models were additonally trained with tiled_dataset. 

### System Requirements
* Google Colab utilizing the High-RAM environment and T4 GPU
* Python 3.11.13
* PyTorch 2.6.0
* Torchvision 0.21.0
* YOLO models were trained with the official Ultralytics package (v8.3.161)
* Faster R-CNN was implemented with Torchvision’s fasterrcnn_resnet50_fpn.

### Dependencies
In order to run the script, the following pakcages must be installed in your Colab environment.
```
!pip install --upgrade git+https://github.com/sunsmarterjie/yolov12.git
!pip install -q supervision flash-attn
```

## Instructions 

1. Save datasets to google drive. Use the following directory:
```
/content/drive/MyDrive/tiled_dataset
/content/drive/MyDrive/train_split_complete_dataset
```

2. To train the model(s), execute function **train_model**:
```
train_model(model = 'YOLOv12', model_size = 's', dataset = 'default', version = '1')
```
**Note**  
Supported models: `YOLOv8`, `YOLOv12`, `Faster R-CNN`  
- **YOLOv8** sizes: `n`, `m`, `l`  
- **YOLOv12** sizes: `s`, `m`, `l`, `x`  
Available datasets: `default`, `tiled`  
The `version` parameter defines the naming convention for saved models.

3. To evaluate the trained model(s), execute function **evalute_model**:
```
evaluate_model(model = 'YOLOv12', model_size = 's', dataset = 'default', version = '1', confidence = 0.25, save_results = True)
```
**Note**: Arguments for evaluate_model is similar to train_model. The argument 'confidence' pertains to the confidence used for predicted bounding boxes. Set argument 'save_results' to True to create a .csv file of results.

4. To evaluate different confidence values, use function **confidence_screeplot**:
```
confidence_screeplot(model = 'YOLOv12', model_size = 's', dataset = 'default', version = '1')
```
## Citations

**Roboflow**:
```
B. Dwyer, J. Nelson, T. Hansen et al., “Roboflow (version 1.0) [soft-
ware],” https://roboflow.com, 2025, computer vision.
```

**YOLOv12**
```
Y. Tian, Q. Ye, and D. Doermann, “Yolov12: Attention-centric real-time
object detectors,” arXiv preprint arXiv:2502.12524, 2025.
```

**YOLOv8**
```
G. Jocher, J. Qiu, and A. Chaurasia, “Ultralytics YOLO,” Jan. 2023.
[Online]. Available: https://github.com/ultralytics/ultralytics
```

**Faster R-CNN**
```
S. Ren, K. He, R. Girshick, and J. Sun, “Faster r-cnn: Towards real-time
object detection with region proposal networks,” IEEE transactions on
pattern analysis and machine intelligence, vol. 39, no. 6, pp. 1137–1149,
2016
```


## License
MIT License - You are free to use and modify the code
