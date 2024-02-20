import matplotlib.pyplot as plt
import os
import cv2 as cv
import json
import numpy as np
import torch
import copy

class Cell:
    def __init__(self, top, left, bottom, right, content=None):
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        self.pos = (top, left), (bottom, right)
        self.content = content
    def __repr__(self):
        if self.content is not None:
            return f"Cell[t={self.top}, l={self.left}, b={self.bottom}, r={self.right}, content='{self.content}']"
        return f"Cell[t={self.top}, l={self.left}, b={self.bottom}, r={self.right}]"
    def overlap(self, other):
        return not (self.right < other.left or
                    self.left > other.right or
                    self.bottom < other.top or
                    self.top > other.bottom)


def load_merge_gt(merge_labels, img_name):
    '''Load merge ground truth for an image'''
    r_gt = np.array(merge_labels[img_name]['rows'])
    c_gt = np.array(merge_labels[img_name]['columns'])
    R_gt = np.array(merge_labels[img_name]['h_matrix'])
    D_gt = np.array(merge_labels[img_name]['v_matrix'])
    return r_gt, c_gt, R_gt, D_gt

def borders(x):
    '''
    Args:
    x -- binary numpy array of shape (N,)
    Returns:
    borders -- numpy array containing the indexes of x where values change from 0 to 1 or vice versa
    '''
    x = x.astype(np.int8)
    changes = np.array([abs(x[i+1] - x[i]) for i in range(len(x)-1)])
    borders = np.where(changes == 1)[0]
    # when first cell is right at the beginning
    if x[0] == 0:
        borders = np.concatenate(([0], borders))
    # when last cell is at the end
    if len(borders) % 2 != 0: # NOTE: check this again
        borders = np.concatenate((borders, [len(x) - 1]))
    return borders

def get_cells(row_borders, col_borders):
    '''
    Args:
    row_borders -- indexes of rows' borders
    col_borders -- indexes of cols' borders
    Returns:
    cells -- list of Cells
    '''
    cells = []
    for i in range(0, len(row_borders), 2):
        for j in range(0, len(col_borders), 2):
            if (i + 1) >= len(row_borders) or (j + 1) >= len(col_borders): 
                continue
            top, left = row_borders[i], col_borders[j]
            bottom, right = row_borders[i+1], col_borders[j+1]
            cells.append(Cell(top, left, bottom, right))
    return cells

def create_pred_matrices(row_borders, col_borders):
    '''
    Returns prediction matrix R and D filled with zeros
    Returns None when the original image has only a row or a column
    '''
    num_rows, num_cols = round(len(row_borders) / 2), round(len(col_borders) / 2)
    # merge RIGHT matrix
    R_pred = np.zeros((num_rows, num_cols-1), dtype=np.uint8) if num_cols > 1 else None
    # merge DOWN matrix
    D_pred = np.zeros((num_rows-1, num_cols), dtype=np.uint8) if num_rows > 1 else None
    return R_pred, D_pred

def get_shape(R, D):
    num_rows, num_cols = None, None
    if R is not None:
        num_rows = R.shape[0]
        num_cols = R.shape[1] + 1
    elif D is not None:
        num_rows = D.shape[0] + 1
        num_cols = D.shape[1]
    else:
        print('ERROR: Both R and D matrices are None')
    return num_rows, num_cols

#######################################################################################
### RULE 1: Merge cells where predicted seperator passes through text #################
#######################################################################################

def id2coord(id, n_cols):
    '''Converts index to (x,y)'''
    x = id // n_cols
    y = id % n_cols
    return x, y

def neighbor_RD(idx, num_rows, num_cols):
    '''Returns indexes for right and down neighbours'''
    idx_right = idx + 1 if (idx + 1) % num_cols != 0 else None
    idx_down = idx + num_cols if (idx + num_cols) < (num_rows * num_cols) else None
    return idx_right, idx_down

# NOTE: these two functions are not quite efficient yet
def check_merge_right(id, id_r, cells, texts_pos):
    (t1, l1), (b1, r1) = cells[id].pos
    (t2, l2), (b2, r2) = cells[id_r].pos
    for text, (l, t, r, b) in texts_pos:
        if l < r1 and l2 < r and (t1 < t < b1 or t1 < b < b1 or (t < t1 and b > b1)):
            return True
    return False

def check_merge_down(id, id_d, cells, texts_pos):
    (t1, l1), (b1, r1) = cells[id].pos
    (t2, l2), (b2, r2) = cells[id_d].pos
    for text, (l, t, r, b) in texts_pos:
        if t < b1 and t2 < b and (l1 < l < r1 or l1 < r < r1 or (l < l1 and r > r1)):
            return True
    return False

def rule1(cells, texts_pos, R_pred, D_pred, verbose=False):
    '''Inplace updates prediction matrices R and D'''
    num_rows, num_cols = get_shape(R_pred, D_pred)
    
    for id, cell in enumerate(cells):
        x, y = id2coord(id, num_cols)
        rn, dn = neighbor_RD(id, num_rows, num_cols)
        if rn and R_pred is not None and check_merge_right(id, rn, cells, texts_pos):
            R_pred[x, y] = 1
            if verbose: print(f'Merge right at cell ({x},{y})')
        if dn and D_pred is not None and check_merge_down(id, dn, cells, texts_pos):
            D_pred[x, y] = 1
            if verbose: print(f'Merge down at cell ({x},{y})')

#######################################################################################
### RULE 2: In the first row (likely header row),           ###########################
###         merge non-blank cells with adjacent blank cells ###########################
#######################################################################################

def is_blank(id, cells, texts_pos):
    (top, left), (bottom, right) = cells[id].pos
    for text, (l, t, r, b) in texts_pos:
        if (top < b and bottom > t) and (left < r and right > l):
            return False
    return True

def rule2(cells, texts_pos, R_pred, D_pred, verbose=False):
    # NOTE: currently only merge right for cells in the first
    #       row if its right neighbour is blank to avoid ambiguity
    num_rows, num_cols = get_shape(R_pred, D_pred)
    for id, cell in enumerate(cells):
        x, y = id2coord(id, num_cols)
        rn, _ = neighbor_RD(id, num_rows, num_cols)
        if x == 0 and rn and R_pred is not None: # first row and has right neighbour
            # if right cell is blank and current cell is non-blank, merge right
            if not is_blank(id, cells, texts_pos) and is_blank(rn, cells, texts_pos):
            # if is_blank(rn, cells, texts_pos):
                R_pred[x, y] = 1
                if verbose: print(f'Merge right at cell ({x},{y})')

#######################################################################################
### RULES END #######################################################################
#######################################################################################
                
# NOTE: might need to check L-shape case
def merge_right(cell, cell_right):
    (t1, l1), (b1, r1) = cell.pos
    (t2, l2), (b2, r2) = cell_right.pos
    assert t1 == t2 and b1 == b2, f"Top and bottom of two cells don't match when merging right: {cell} & {cell_right}"
    assert l1 <= r2, "Cell to merge right is to the left of current cell: {cell} & {cell_right}"
    return Cell(t1, l1, b1, r2)

def merge_down(cell, cell_down):
    (t1, l1), (b1, r1) = cell.pos
    (t2, l2), (b2, r2) = cell_down.pos
    assert l1 == l2 and r1 == r2, f"Left and right boundaries of two cells don't match when merging down: {cell} & {cell_down}"
    assert b1 <= t2, "Cell to merge down is not just below the current cell: {cell} & {cell_down}"
    return Cell(t1, l1, b2, r1)

### THIS PART MIGHT POTENTIALLY BE IMPROVED ###

def _merge_overlap(cells):
    """
    Merge overlapping cells in a list.
    Params:
    - cells: List of Cell objects.
    Returns:
    - List of merged Cell objects.
    """
    merged_cells = []

    for i, cell in enumerate(cells):
        is_merged = False

        for j, merged_cell in enumerate(merged_cells):
            if cell.overlap(merged_cell):
                # Merge overlapping cells
                merged_cells[j] = Cell(min(cell.top, merged_cell.top),
                                       min(cell.left, merged_cell.left),
                                       max(cell.bottom, merged_cell.bottom),
                                       max(cell.right, merged_cell.right))
                is_merged = True
                break

        if not is_merged:
            # Add the non-overlapping cell to the merged_cells list
            merged_cells.append(cell)

    return merged_cells

def merge_cells(cells, R, D, verbose=False):
    merged_cells = []
    cells = copy.deepcopy(cells)
    n_rows, n_cols = get_shape(R, D)

    if len(cells) != n_rows * n_cols:
        print("ERROR: Shape of R and D don't match the number of cells")
        return None
    
    for i, cell in enumerate(cells):
        x, y = id2coord(i, n_cols)
        rn_id, dn_id = neighbor_RD(i, n_rows, n_cols)

        if (rn_id is not None) and (R is not None) and R[x, y] == 1:
            if verbose: print(f'Merge right at cell ({x},{y})')
            merged_cells.append(merge_right(cell, cells[rn_id]))
            
        elif (dn_id is not None) and (D is not None) and D[x, y] == 1:
            if verbose: print(f'Merge down at cell ({x},{y})')
            merged_cells.append(merge_down(cell, cells[dn_id]))
        
        else:
            merged_cells.append(cell)

    return _merge_overlap(merged_cells)

######################################################

def IoU(cell_1, cell_2):
    (t1, l1), (b1, r1) = cell_1.pos
    (t2, l2), (b2, r2) = cell_2.pos

    # Check for invalid bounding boxes
    if t1 >= b1 or l1 >= r1 or t2 >= b2 or l2 >= r2:
        # print('Coordinates of the cells are invalid')
        return -1

    inner_top, inner_left = max(t1, t2), max(l1, l2)
    inner_bot, inner_right = min(b1, b2), min(r1, r2)

    if inner_bot >= inner_top and inner_right >= inner_left:
        inner_area = (inner_bot - inner_top) * (inner_right - inner_left)
        area_1 = (b1 - t1) * (r1 - l1)
        area_2 = (b2 - t2) * (r2 - l2)

        # Use floating-point division
        union_area = area_1 + area_2 - inner_area
        return inner_area / union_area if union_area > 0 else 0.0
    else:
        return 0.0
    
def eval(cells_pred, cells_label, threshold=0.7, img_name=None):
    '''Returns F1, recall, and precision score'''
    n_correct_preds = 0
    n_preds, n_true = len(cells_pred), len(cells_label)
    flag_error = False

    # Count true positives, false positives, and false negatives
    for pred_box in cells_pred:
        max_iou = 0
        for true_box in cells_label:
            iou = IoU(pred_box, true_box)
            if iou == -1: flag_error = True
            if iou > max_iou:
                max_iou = iou

        if max_iou >= threshold:
            n_correct_preds += 1

    # Calculate precision, recall, and F1 score
    precision = n_correct_preds / n_preds if n_preds > 0 else 0.0
    recall = n_correct_preds / n_true if n_true > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    if flag_error is True and img_name is not None:
        # print(f'Image {img_name} has cells with wrong coordinates')
        return f1, recall, precision, img_name
    return f1, recall, precision, None