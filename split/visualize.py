import torch
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from dataset.dataset import ImageDataset
from modules.split_modules import SplitModel

def visualize_split(model: SplitModel, dataset: ImageDataset, id: int, save_dir=None):
    model.eval()
    img, label = dataset[id]
    r, c = model(img.unsqueeze(0))
    
    # format predictions
    r = r[-1]>0.5
    c = c[-1]>0.5
    c = c.cpu().detach().numpy()
    r = r.cpu().detach().numpy()
    r_im = r.reshape((-1,1)) * np.ones((r.shape[0],c.shape[0]))
    c_im = c.reshape((1,-1)) * np.ones((r.shape[0],c.shape[0]))
    mask = cv.bitwise_or(r_im,c_im)

    img_np = (img * 255).numpy().transpose(1,2,0).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    mask_inv = cv.bitwise_not(mask)

    res = cv.bitwise_and(img_np, img_np, mask=mask_inv)
    res2 = cv.addWeighted(img_np, 0.5, res, 0.5, 0)
    if save_dir:
        save_path = os.path.join(save_dir, f'split_pred_{id}.png')
        cv.imwrite(save_path, cv.cvtColor(res2, cv.COLOR_RGB2BGR))
    else:
        plt.figure(figsize=(10,10))
        plt.imshow(res2)