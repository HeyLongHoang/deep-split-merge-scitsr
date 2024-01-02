# Helper functions
import matplotlib.pyplot as plt
import os
import cv2 as cv
import json
import numpy as np
import torch

def view_imgs(imgs, titles=None):
    fig = plt.figure(figsize=(7, 7))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i + 1)
        if titles is not None:
            plt.title(titles[i])
        if img.ndim == 3:
            if img.shape[-1] == 3:
                plt.imshow(img)
            else:
                plt.imshow(img[..., 0], cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

def load_image(img_path):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def save_img(save_path, img):
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(save_path, img)

def load_txt(txt_path):
    with open(txt_path, 'r') as file:
        names = [line.strip() for line in file]
    return names

def path2name(img_path):
    return os.path.splitext(os.path.basename(img_path))[0]

def load_json(json_path):
    with open(json_path, 'r') as f:
        info = json.load(f)
    return info

def vis_split(img, r, c, color='blue'):
    '''Visualize split prediction'''
    if color == 'red': theme = (255,0,0)
    elif color == 'green': theme = (0,255,0)
    else: theme = (0,0,255)
    r_im = r.reshape((-1,1)) * np.ones((r.shape[0],c.shape[0]))
    c_im = c.reshape((1,-1)) * np.ones((r.shape[0],c.shape[0]))
    bg = np.maximum(r_im, c_im)
    bg_img = np.concatenate(
        [bg[...,np.newaxis] * color for color in theme],
        axis=2
    )
    bg_img[np.all(bg_img == (0, 0, 0), axis=-1)] = (255,255,255)
    split_img = ((0.5 * img) + (0.5 * bg_img)).astype("uint8")
    split_img = np.minimum(split_img, img)
    view_imgs([split_img])
    return split_img

def prep_image(img: np.array):
    '''Turn numpy array into torch tensor ready to process by model'''
    return torch.tensor(img / 255.).permute(2,0,1).float().unsqueeze(0)

def process_split_results(r_pred: torch.tensor, c_pred: torch.tensor):
    r = r_pred[-1] > 0.5
    c = c_pred[-1] > 0.5
    c = c.cpu().detach().numpy().astype(np.uint8)
    r = r.cpu().detach().numpy().astype(np.uint8)
    return r, c

def refine_split_results(x, threshold=2, adj_threshold=5):
    updated_x = x.copy()

    # Find indices of 1's in the array
    ones_indices = np.where(updated_x == 1)[0]

    # Iterate through the 1's indices and check for merging conditions
    i = 0
    while i < len(ones_indices):
        start_idx = ones_indices[i]
        end_idx = start_idx + threshold - 1

        # Check if there is another group of 1's within adj_threshold to the left or right
        if i + threshold < len(ones_indices) and ones_indices[i + threshold] - end_idx <= adj_threshold:
            end_idx = ones_indices[i + threshold]

            # Update all 1's in the range between the two groups to 1
            updated_x[start_idx:end_idx + 1] = 1

            # Move to the next unprocessed index
            i += threshold
        else:
            # Check if the merging condition is satisfied at the beginning of the array
            if start_idx - threshold >= 0 and start_idx - ones_indices[i - threshold] <= adj_threshold:
                start_idx = ones_indices[i - threshold]

                # Update all 1's in the range between the two groups to 1
                updated_x[start_idx:end_idx + 1] = 1
            else:
                updated_x[start_idx:end_idx + 1] = 0

            i += 1

    return updated_x