# -*- coding: utf-8 -*-
# Author: craig.li(solitaire10@163.com)
# Image dataset for training and testing Split and Merge models(ICDAR 2019 :Deep Splitting and Merging for Table Structure Decomposition).

import cv2
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
  """Image Dataset"""

  def __init__(self, img_dir, labels_dict, output_width, scale=0.5,
               min_width=40, mode='split', suffix='.png', returns_image_name=False):
    """
    Initialization of the dataset

    Args:
      img_dir(str): The directory of images
      labels_dict(dict): A dictionary stores ids of images and
          corresponding ground truth, which are two vectors with
          w and h elements, where w is the width of image and h
          is the height of image, indicates the probability of
          there is a line in that row or column.
      output_width(int): Defines the width of the output tensor,
      scale(float): The scale of resizing image.
      min_width(int): Specifies minimal width of resizing image.
      mode(str): The model should be one of 'split' and 'merge'
    """
    self.labels_dict = labels_dict
    self.ids = list(labels_dict.keys())
    self.nSamples = len(self.ids)
    self.img_dir = img_dir
    self.output_width = output_width
    self.scale = scale
    self.min_width = min_width
    self.mode = mode
    self.suffix = suffix
    self.returns_image_name = returns_image_name

  def __len__(self):
    return self.nSamples

  def __getitem__(self, index):
    assert index <= len(self), 'index range error'
    id = self.ids[index]
    if self.suffix == '.npy':
      img = np.load(os.path.join(self.img_dir, id + self.suffix))
    elif self.suffix == '.jpg':
      img = Image.open(os.path.join(self.img_dir, id + self.suffix))
      img = np.array(img)
    elif self.suffix == '.png':
      img = cv2.imread(os.path.join(self.img_dir, id + self.suffix))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = np.transpose(img, (2,0,1))

    c, h, w = img.shape
    new_h = int(self.scale * h) if int(self.scale * h) > self.min_width \
                                else self.min_width
    new_w = int(self.scale * w) if int(self.scale * w) > self.min_width \
                                else self.min_width
    img = np.array(
      [cv2.resize(img[i], (new_w, new_h), interpolation=cv2.INTER_AREA) 
       for i in range(c)]
    )
    img_array = np.array(img) / 255.

    if self.mode == 'merge':
      labels = self.labels_dict[id]
      rows = labels['rows']
      columns = labels['columns']
      h_matrix = labels['h_matrix']
      v_matrix = labels['v_matrix']

      img_tensor = torch.from_numpy(img_array).type(torch.float)
      row_label = torch.from_numpy(np.array(h_matrix)).type(torch.float)
      column_label = torch.from_numpy(np.array(v_matrix)).type(torch.float)

      return img_tensor, (row_label, column_label), (rows, columns)
    
    else: # 'split'
      labels = self.labels_dict[id]
      row_label = labels['rows']
      column_label = labels['columns']

      # resize ground truth to proper size
      width = int(np.floor(new_w / self.output_width))
      height = int(np.floor(new_h / self.output_width))
      row_label = np.array([row_label]).T * np.ones((len(row_label), width))
      column_label = np.array(column_label) * np.ones((height, len(column_label)))

      row_label = np.array(row_label, dtype=np.uint8)
      column_label = np.array(column_label, dtype=np.uint8)

      row_label = cv2.resize(row_label, (width, new_h))
      column_label = cv2.resize(column_label, (new_w, height))

      row_label = row_label[:, 0]
      column_label = column_label[0, :]

      img_tensor = torch.from_numpy(img_array).type(torch.float)
      row_label = torch.from_numpy(row_label).type(torch.float)
      column_label = torch.from_numpy(column_label).type(torch.float)

      if self.returns_image_name:
        return img_tensor, (row_label, column_label), id
      return img_tensor, (row_label, column_label)
