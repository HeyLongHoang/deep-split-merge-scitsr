import os
import cv2 as cv
import torch
import glob
import json
import math
from data_utils.utils import *
from modules.split_modules import SplitModel
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_weight', help='Path to weights for Split model')
    parser.add_argument('--pred_dir', help='Path to either train, val, or test directory')
    parser.add_argument('--save_dir', help='Path to directory for saving prediction results')
    parser.add_argument('--img_names', help='Path to .txt file containing image names, one name per line')
    args = parser.parse_args()

    img_names = load_txt(args.img_names) if args.img_names is not None else None
    split_model = load_split_model(args.model_weight)
    # get_predictions(split_model, args.pred_dir, args.save_dir, img_names)
    get_eval(args.save_dir)

def load_split_model(model_weight_path):
    model = SplitModel(3)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        model.load_state_dict(torch.load(model_weight_path))
    else:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    return model

def get_predictions(model, pred_dir, save_dir, img_names=None):
    '''
    Args:
    model -- Split model
    pred_dir -- directory containing two sub-directories called 'img' and 'label'
                where 'img' sub-dir contains images and 'label' sub-dir contains 
                two ground truth files called 'split_label.json' and 'merge_label.json'
    save_dir -- directory to save prediction results
    img_names -- (optional) list of names of images inside pred_dir/img to make predictions
                if None then give predictions for all images inside pred_dir/img
    '''
    model.eval()
    img_dir = os.path.join(pred_dir, 'img')
    split_labels = load_json(os.path.join(pred_dir, 'label', 'split_label.json'))
    if img_names == None:
        img_paths = glob.glob(os.path.join(img_dir, '*.png'))
    else:
        img_paths = [os.path.join(img_dir, img_name + '.png') for img_name in img_names]

    print('Getting predictions...')
    pred_saved = 0

    for img_path in img_paths:
        res = {}
        if not os.path.isfile(img_path): 
            continue

        img = load_image(img_path)
        img_name = path2name(img_path)
        img_save_dir = os.path.join(save_dir, img_name)
        os.makedirs(img_save_dir, exist_ok=True)
        
        # prediction
        img_ready = prep_image(img)
        with torch.no_grad():
            r_pred, c_pred = model(img_ready)
        r_pred, c_pred = process_split_results(r_pred, c_pred)
        img_split = vis_split(img, r_pred, c_pred, view=False)

        # ground truth
        gt = split_labels[img_name]
        r_gt = np.array(gt['rows'], dtype=np.uint8)
        c_gt = np.array(gt['columns'], dtype=np.uint8)
        img_gt = vis_split(img, r_gt, c_gt, 'red', view=False)

        save_img(os.path.join(img_save_dir, img_name + '.png'), img)
        save_img(os.path.join(img_save_dir, img_name + '_pred.png'), img_split)
        save_img(os.path.join(img_save_dir, img_name + '_gt.png'), img_gt)

        res['row'] = eval3(r_pred, r_gt)
        res['col'] = eval3(c_pred, c_gt)
        with open(os.path.join(img_save_dir, 'result.json'), 'w') as f:
            json.dump(res, f)

        pred_saved += 1

    print(f'Done saving predictions for {pred_saved}/{len(img_paths)} images!')

def get_eval(res_dir, top_n=5):
    '''
    Args:
    res_dir -- directory containing prediction results, where each sub-directory (image name) 
                contains a result.json file with the following format, for example
                {
                    "row": {"precision": 0.9, "recall": 0.7, "f1": 0.8}, 
                    "col": {"precision": 0.4, "recall": 0.9, "f1": 0.6}
                }
    '''
    json_paths = glob.glob(os.path.join(res_dir, '*', 'result.json'))
    r_precs, r_recs, r_f1s = [], [], []
    c_precs, c_recs, c_f1s = [], [], []
    print(f'Calculating average scores for {len(json_paths)} images...')

    for json_path in json_paths:
        res = load_json(json_path)
        row, col = res['row'], res['col']
        r_precs.append(row['precision']); r_recs.append(row['recall']); r_f1s.append(row['f1'])
        c_precs.append(col['precision']); c_recs.append(col['recall']); c_f1s.append(col['f1'])

    row_stats = {
        'precision': np.mean(r_precs).tolist(),
        'recall': np.mean(r_recs).tolist(),
        'f1': np.mean(r_f1s).tolist()
    }

    col_stats = {
        'precision': np.mean(c_precs).tolist(),
        'recall': np.mean(c_recs).tolist(),
        'f1': np.mean(c_f1s).tolist()
    }
    res = {'row': row_stats, 'col': col_stats}
    
    write_json(os.path.join(res_dir, 'stats.json'), res)
    print('Done!')

if __name__ == '__main__':
    main()
