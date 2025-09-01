# -*- coding: utf-8 -*-
"""
# Package Dependencies and Imports
"""

!pip install --upgrade git+https://github.com/sunsmarterjie/yolov12.git
!pip install -q supervision flash-attn

import os
import torch
import csv
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_iou
from ultralytics import YOLO
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

"""# CONFIG"""

CONFIG = {
    'epochs': 100,
    'batch_size': 16,
    'image_size': 960,
    'save_path': '/content/drive/MyDrive/Leaf_Count_Models',
    'save_path_csv': '/content/drive/MyDrive/Model_MAE.csv'
}

"""# Training Function"""

def train_model(model = 'YOLOv12', model_size = 's', dataset = 'tiled', version = '1'):
  match model:
    case 'YOLOv12':
      if model_size not in {'s', 'm', 'l', 'x'}:
        raise ValueError(f'Model size {model_size} is not available for YOLOv12. Please choose from the following: ("s", "m", "l", "x")')
      model_name = f'{model.lower()}{model_size}.yaml'
      model_train = YOLO(model_name)
    case 'YOLOv8':
      if model_size not in {'n', 'm', 'l'}:
        raise ValueError(f'Model size {model_size} is not available for YOLov8. Please choose from the following: ("n", "m", "l")')
      model_name = f'{model.lower()}{model_size}.pt'
      model_train = YOLO(model_name)
    case _:
      raise ValueError(f'Choose among these choices: "YOLOv8", "YOLOv12", "Faster R-CNN"')

  match dataset:
    case 'tiled':
      HOME = '/content/drive/MyDrive/tiled_dataset'
    case 'default':
      HOME = '/content/drive/MyDrive/train_split_complete_dataset'
    case _:
      raise ValueError(f'Dataset {dataset} is not available. Please choose from the following: "tiled", "default"')

  model_train.train(data = f'{HOME}/data.yaml',
                    epochs = CONFIG['epochs'],
                    batch = CONFIG['batch_size'],
                    imgsz = CONFIG['image_size'],
                    save = True,
                    save_period = 10,
                    project = CONFIG['save_path'],
                    name = f'{model.lower()}{model_size}_v{version}_checkpoint',
                    exist_ok = True)

  metrics = model_train.val()
  print(metrics)

"""# Evaluation Function"""

def calculate_iou(pred_boxes, gt_boxes):
  if len(pred_boxes) == 0 or len(gt_boxes) == 0:
    return 0.0
  iou_matrix = box_iou(pred_boxes, gt_boxes)
  return iou_matrix.max(dim=1)[0].mean().item()

def load_ground_truth(label_path, img_width, img_height):
  boxes = []
  if not os.path.exists(label_path):
    return torch.empty((0, 4))

  with open(label_path, 'r') as f:
    for line in f:
      parts = line.strip().split()
      if len(parts) != 5:
        continue
      cls, cx, cy, w, h = map(float, parts)
      x1 = (cx - w / 2) * img_width
      y1 = (cy - h / 2) * img_height
      x2 = (cx + w / 2) * img_width
      y2 = (cy + h / 2) * img_height
      boxes.append([x1, y1, x2, y2])

  return torch.tensor(boxes)

def evaluate_model(model, model_size, dataset, version, confidence = 0.25, save_results = False):
  postfix = '/weights/best.pt'
  save_path = CONFIG['save_path']
  model_path = f'{save_path}/{model.lower()}{model_size}_v{version}_checkpoint{postfix}'

  match dataset:
    case 'tiled':
      HOME = '/content/drive/MyDrive/tiled_dataset'
    case 'default':
      HOME = '/content/drive/MyDrive/train_split_complete_dataset'
    case _:
      raise ValueError(f'Dataset {dataset} is not available. Please choose from the following: "tiled", "default"')

  img_val = f'{HOME}/images/valid'
  lbl_val = f'{HOME}/labels/valid'

  model = YOLO(model_path)
  iou_scores = []
  rows = []
  mae_values = []
  mape_values = []
  mse_values = []
  for img_file in os.listdir(img_val):
    if not img_file.endswith(('.jpg', '.png', '.jpeg')):
      continue

    image_path = os.path.join(img_val, img_file)
    label_path = os.path.join(lbl_val, os.path.splitext(img_file)[0] + '.txt')

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    gt_boxes = load_ground_truth(label_path, width, height)

    if not os.path.exists(label_path):
      gt_count = 0
    else:
      with open(label_path, 'r') as f:
        gt_count = len([line for line in f if line.strip()])

    results = model.predict(
        source = image_path,
        conf = confidence
    )
    pred = results[0]
    pred_boxes = pred.boxes.xyxy.cpu() if pred.boxes is not None else torch.empty((0, 4))

    # Calculate IOU
    iou = calculate_iou(pred_boxes, gt_boxes)
    iou_scores.append(iou)

    # Calculate MSE, MAE, MAPE
    pred_count = len(pred.boxes) if pred.boxes else 0

    sq_err = (gt_count - pred) ** 2
    mse_values.append(sq_err)

    mae = abs((gt_count - pred)/gt_count)
    mae_values.append(mae)

    mape = abs((gt_count - pred_count)/gt_count)
    mape_values.append(mape)

    rows.append([img_file, gt_count, pred_count, sq_err])

  if save_results == True:
    with open(CONFIG['save_path_csv'], 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writero(['image_name', 'ground_truth', 'predicted', 'squared_error'])
      writer.writerows(rows)

  avg_iou = np.mean(iou_scores)
  avg_mse = sum(mse_values) / len(mse_values)
  avg_mae = sum(mae_values) / len(mae_values)
  avg_mape = (sum(mape_values) / len(mape_values)) * 100
  print(f'Average IoU: {avg_iou: .4f} over {len(iou_scores)} validation images')
  print(f'Average MSE: {avg_mse: .4f}')
  print(f'Average MAE: {avg_mae: .4f}')
  print(f'Average MAPE: {avg_mape: .4f}')

  return avg_iou, avg_mse, avg_mae, avg_mape

def plot_results(conf, result, result_text):
  plt.figure(figsize = (12, 6), dpi = 100)
  plt.plot(conf, result, marker = 'o', linestyle = '-')

  plt.xticks(
      ticks = conf,
      labels = [f'{x:.2f}' for x in conf],
      rotation = 45,
      ha = 'right'
  )
  for x, y in zip(conf, result):
    plt.annotate(
        f'{y:.2f}',
        xy = (x, y),
        xytext = (0, 5),
        textcoords = 'offset points',
        fontsize = 8,
        ha = 'center',
        va = 'bottom'
    )

  plt.xlabel('Confidence')
  plt.ylabel(result_text)
  plt.title('Scree Plot for Confidence Value')
  plt.show()

def confidence_screeplot(model, model_size, dataset, version):
  confs = np.arange(0, 1.05, 0.05)
  mses = []
  ious = []
  maes = []
  mapes = []
  for conf in confs:
    iou, mse, mae, mape = evaluate_model(model = model,
                                         model_size = model_size,
                                         dataset = dataset,
                                         version = version,
                                         confidence = conf)
    mses.append(mse)
    ious.append(iou)
    maes.append(mae)
    mapes.append(mape)
  plot_results(confs, ious, "IOU")
  plot_results(confs, mses, "MSE")
  plot_results(confs, mae, "MAE")
  plot_results(confs, mape, "MAPE")