import argparse
import os
import json
import cv2 as cv
from pdf2image import convert_from_path
import numpy as np
import matplotlib.pyplot as plt

def view_imgs(imgs):
    fig = plt.figure(figsize=(7,7))
    for i, img in enumerate(imgs):
        plt.subplot(1, len(imgs), i+1)
        if img.ndim == 3:
            if img.shape[-1] == 3:
                plt.imshow(img)
            else:
                plt.imshow(img[..., 0], cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()

def path2name(img_path):
    return os.path.splitext(os.path.basename(img_path))[0]

def get_table_pos(pdf_path, display=False):
    '''Get table position inside pdf file'''
    imgs_pdf = convert_from_path(pdf_path, 150)
    assert len(imgs_pdf) == 1, 'There is more than one pdf image'
    img_pdf = np.array(imgs_pdf[0])
    gray = cv.cvtColor(img_pdf, cv.COLOR_RGB2GRAY)
    thresh, gray = cv.threshold(gray, 200, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    coords = cv.findNonZero(gray)
    x, y, w, h = cv.boundingRect(coords)
    left, top, right, bottom = x - 3, y - 3, x + w + 3, y + h + 3
    if display:
        top_left = (round(left), round(top))
        bottom_right = (round(right), round(bottom))
        cv.rectangle(img_pdf, top_left, bottom_right, (0, 0, 255), 3)
        view_imgs([img_pdf])
        print('Shape: ' + str(img_pdf.shape))
    return left, top, right, bottom   

def coord_pdf2img(x, y, left, top):
    ratio = 1754 / 842
    new_x = x * ratio - left
    new_y = 1754 - y * ratio - top
    return new_x, new_y

def read_chunk(chunk_path, left, top, display=False):
    Chunks = []
    with open(chunk_path) as f:
      data = json.load(f)
      chunks = data['chunks']
      for c in chunks:
          pos = c['pos']
          l,r,t,b = int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])
          if display: print('OG       :', l,t,r,b, '--', c['text'])
          l,t = coord_pdf2img(l,t, left, top)
          r,b = coord_pdf2img(r,b, left, top)
          if display: print('Converted:', int(l), int(t), int(r), int(b))
          if t > b: t, b = b, t
          Chunks.append((c['text'], [int(l), int(t)-5, int(r), int(b)]))
    return Chunks   

def main():
    '''
    NOTE: During reading images, there will be some corrupted ones. Because of that,
            some errors will appear but the program will still continue to run, so 
            please ignore that!
    '''
    p = argparse.ArgumentParser()
    p.add_argument('--scitsr_dir',
                   type=str,
                   help='Path to train or test folder of SciTSR dataset')
    p.add_argument('--chunk_json', 
                   type=str, 
                   help='Path the the ground truth file')
    
    # Get inputs from command call
    arg = p.parse_args()
    img_dir = os.path.join(arg.scitsr_dir, 'img')
    pdf_dir = os.path.join(arg.scitsr_dir, 'pdf')
    
    # Create json ground truth folder
    json_dir = os.path.dirname(arg.chunk_json)
    os.makedirs(json_dir, exist_ok=True)
    
    # Get images of tabels
    imgs_paths = [os.path.join(img_dir, p) for p in os.listdir(img_dir)]
    valid_paths = [p for  p in imgs_paths if cv.imread(p) is not None]
    
    LABELS = {}
    for i, img_path in enumerate(valid_paths):
        # Get table position inside pdf file
        img_pdf_path = img_path.replace('png', 'pdf').replace('img', 'pdf')
        left, top, right, bottom = get_table_pos(img_pdf_path)
        
        # Get chunk info (position of cells in the table)
        img_chunk_path = img_path.replace('png', 'chunk').replace('img', 'chunk')
        chunks = read_chunk(img_chunk_path, left, top)

        img_name = path2name(img_path)
        LABELS[img_name] = chunks

        if i % 100 == 0:
            img_draw = cv.imread(img_path)
            for chunk in chunks:
                pos = chunk[1]
                left_top = (pos[0], pos[1])
                bottom_right = (pos[2], pos[3])
                cv.rectangle(img_draw, left_top, bottom_right, (0, 0, 255), 1)
            # view_imgs([img_draw])
            print(f'- Processed {i+1} images up until now!')

    # Dump chunk info
    with open(arg.chunk_json, 'w') as f:
        json.dump(LABELS, f)

if __name__ == '__main__':
    main()