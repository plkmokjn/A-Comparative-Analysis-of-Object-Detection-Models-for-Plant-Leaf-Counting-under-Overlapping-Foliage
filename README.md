# A-Comparative-Analysis-of-Object-Detection-Models-for-Plant-Leaf-Counting-under-Overlapping-Foliage-Conditions

Comparative Analysis of three foundational object detection models: YOLOv8, YOLOv12, and Faster R-CNN for leaf cointing in dense foliage. Evaluated on all performance metric: mAP50, mAP50-95, IOU, MAE, and MAPE.

## Overview



### Datasets
The datasets used in this project is hosted on Google Drive. You can access the data via the following link:
- [Raw Dataset](https://drive.google.com/drive/folders/1SqZVu7KqB2eOgOt82ETEhWRWwm2cNwmN?usp=sharing)
- [train_split_complete_dataset](https://drive.google.com/drive/folders/13MaM3n2fJA5EUR2EBFwm7Byr42MP6Ge3?usp=sharing)

### System Requirements
* Google Colab

### Dependencies
In order to run the scripts, the following pakcages must be installed in your Colab environment.
```
!pip install --upgrade git+https://github.com/sunsmarterjie/yolov12.git
!pip install -q supervision flash-attn
```

## Instructions 

"""# Instructions
1. Save datasets to google drive. Use the following directory:
```
/content/drive/MyDrive/tiled
/content/drive/MyDrive/train_split_complete_dataset
```
2. To train model, execute function **train_model**:
```
train_model(model = 'YOLOv12', model_size = 's', dataset = 'default', version = '1')
```
**Note**: Possible models are 'YOLOv8', 'YOLOv12', 'Faster R-CNN'. Possible sizes varies for each YOLO model. YOLOv8 has 'n', 'm', 'l'. YOLOv12 has 's', 'm', 'l', 'x'. There are two datasets available: 'default' and 'tiled'. version is for naming convention for saving models.

3. To evaluate trained models, execute function **evalute_model**:
```
evaluate_model(model = 'YOLOv12', model_size = 's', dataset = 'default', version = '1', confidence = 0.25, save_results = True)
```
**Note**: Arguments for evaluate_model is similar to train_model. The argument 'confidence' pertains to the confidence used for predicted bounding boxes. Set argument 'save_results' to True to create a .csv file of results.

4. To evaluate different confidence values, use function **confidence_screeplot**:
```
confidence_screeplot(model = 'YOLOv12', model_size = 's', dataset = 'default', version = '1')
```
"""










## License
MIT License - You are free to use and modify the code
