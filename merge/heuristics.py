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
        self.is_blank = None

    def __repr__(self):
        if self.content is not None:
            return f"Cell[t={self.top}, l={self.left}, b={self.bottom}, r={self.right}, content='{self.content}']"
        return f"Cell[t={self.top}, l={self.left}, b={self.bottom}, r={self.right}]"
    
    def overlap(self, other):
        return not (self.right < other.left or
                    self.left > other.right or
                    self.bottom < other.top or
                    self.top > other.bottom)
    
    def update_blank(self, texts_pos):
        '''Update whether the cell is blank or not'''
        self.is_blank = True

        for text, (l, t, r, b) in texts_pos:
            if (self.top < b and self.bottom > t) and (self.left < r and self.right > l):
                self.is_blank = False
                break  # If overlap is found, no need to check further

        # for content, (text_l, text_t, text_r, text_b) in texts_pos:
        #     if (text_l >= self.left or self.left - text_l <= margin) and \
        #         (text_t >= self.top or self.top - text_t <= margin) and \
        #         (text_r <= self.right or text_r - self.right <= margin) and \
        #         (text_b <= self.bottom or text_b - self.bottom <= margin) \
        #     :
        #         self.is_blank = False
        #         break

def load_split_gt(split_labels, img_name):
    '''Load split ground truth for an image'''
    r_gt = np.array(split_labels[img_name]['rows'])
    c_gt = np.array(split_labels[img_name]['columns'])
    return r_gt, c_gt

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
        print('N. rows:', num_rows)
        print('N. cols:', num_cols)
    return num_rows, num_cols

def neighbor_top(idx, num_rows, num_cols):
    '''Returns index for the top neighbor'''
    idx_top = idx - num_cols if idx >= num_cols else None
    return idx_top

def neighbor_left(idx, num_rows, num_cols):
    '''Returns index for the left neighbor'''
    idx_left = idx - 1 if idx % num_cols != 0 else None
    return idx_left

def neighbor_right(idx, num_rows, num_cols):
    '''Returns index for the right neighbor'''
    idx_right = idx + 1 if (idx + 1) % num_cols != 0 else None
    return idx_right

def neighbor_down(idx, num_rows, num_cols):
    '''Returns index for the down neighbor'''
    idx_down = idx + num_cols if (idx + num_cols) < (num_rows * num_cols) else None
    return idx_down


#######################################################################################
### RULE 1: Merge cells where predicted seperator passes through text #################
#######################################################################################

def id2coord(id, n_cols):
    '''Converts index to (x,y)'''
    x = id // n_cols
    y = id % n_cols
    return x, y

def coord2id(x, y, n_cols):
    '''Converts (x, y) to index'''
    return x * n_cols + y

def neighbor_RD(idx, num_rows, num_cols):
    '''Returns indexes for right and down neighbours'''
    idx_right = idx + 1 if (idx + 1) % num_cols != 0 else None
    idx_down = idx + num_cols if (idx + num_cols) < (num_rows * num_cols) else None
    return idx_right, idx_down

# NOTE: these two functions are not quite efficient yet
def check_merge_right(id, id_r, cells, texts_pos, margin_perc=0.5, margin_max=5):
    if cells[id].is_blank is None:
        cells[id].update_blank(texts_pos)
    if cells[id_r].is_blank is None:
        cells[id_r].update_blank(texts_pos)

    (t1, l1), (b1, r1) = cells[id].pos
    (t2, l2), (b2, r2) = cells[id_r].pos
    for text, (l, t, r, b) in texts_pos:
        # apply margin to the text bounding box when one of the cells is blank
        margin = min(margin_max, int(margin_perc * abs(r - l))) \
            if cells[id_r].is_blank \
            else 0      
        if (l - margin) < r1 and \
            l2 < (r + margin) and \
            (t1 < t < b1 or t1 < b < b1 or (t < t1 and b > b1)):
            return True
    return False

def check_merge_down(id, id_d, cells, texts_pos, margin_perc=0.5, margin_max=5):
    # update cell blank information upfront to remove complexity
    if cells[id].is_blank is None:
        cells[id].update_blank(texts_pos)
    if cells[id_d].is_blank is None:
        cells[id_d].update_blank(texts_pos)

    (t1, l1), (b1, r1) = cells[id].pos
    (t2, l2), (b2, r2) = cells[id_d].pos
    for text, (l, t, r, b) in texts_pos:
        # apply margin to the text bouding box when one of the cells is blank
        margin = min(margin_max, int(margin_perc * abs(b - t))) \
            if cells[id_d].is_blank \
            else 0
        if (t - margin) < b1 and \
            t2 < (b + margin) and \
            (l1 < l < r1 or l1 < r < r1 or (l < l1 and r > r1)):
            return True
    return False

def rule1(cells, texts_pos, R_pred, D_pred, verbose=False, **kwargs):
    '''Inplace updates prediction matrices R and D'''
    num_rows, num_cols = get_shape(R_pred, D_pred)
    
    for id, cell in enumerate(cells):
        x, y = id2coord(id, num_cols)
        rn, dn = neighbor_RD(id, num_rows, num_cols)
        if rn and R_pred is not None and check_merge_right(id, rn, cells, texts_pos, **kwargs):
            R_pred[x, y] = 1
            if verbose: print(f'Merge right at cell ({x},{y})')
        if dn and D_pred is not None and check_merge_down(id, dn, cells, texts_pos, **kwargs):
            D_pred[x, y] = 1
            if verbose: print(f'Merge down at cell ({x},{y})')

#######################################################################################
### RULE 2: In the first row (likely header row),           ###########################
###         merge non-blank cells with adjacent blank cells ###########################
#######################################################################################

def has_col_sep_between(col1, col2, row, cells, image, n_cols):
    img = invert_threshold(image)
    cell1 = cells[coord2id(row, col1, n_cols)]
    cell2 = cells[coord2id(row, col2, n_cols)]
    assert cell1.top == cell2.top and cell1.bottom == cell2.bottom, 'The two cells should have the same top and bottom coordinates'

    if cell1.right < cell2.left:
        area = img[cell1.top : cell1.bottom, cell1.right : cell2.left]
    elif cell2.right < cell1.left:
        area = img[cell1.top : cell1.bottom, cell2.right : cell1.left]
    else:
        print('The two cells overlap', cell1, cell2)
        return None
    has_sep = (area.mean(axis=0).max() == 1.0)
    return has_sep
            
def find_nearest_non_blank_col(cells, row, col, texts_pos, image, n_cols):
    # update blank info of cells in the 1st row if necessary
    for c in range(0, n_cols):
        id = coord2id(row, c, n_cols) # id of cells in the first row
        if cells[id].is_blank is None:
            cells[id].update_blank(texts_pos)

    # search for the nearest non-blank cell in the first row
    nearest_col = -1
    best_dist = 100000
    for c in range(0, n_cols):
        if cells[coord2id(row, c, n_cols)].is_blank or has_col_sep_between(c, col, row, cells, image, n_cols):
            continue
        dist = abs(col - c)
        if dist < best_dist:
            nearest_col = c
            best_dist = dist
    return nearest_col

def rule2(cells, texts_pos, image, R_pred, verbose=False, only_first_row=True):
    if R_pred is None: return
    n_rows, n_cols = R_pred.shape[0], R_pred.shape[1]+1

    if only_first_row:
        rows = [0]
    else:
        rows = range(n_rows)

    # check conditions and merge
    for row in rows:
        for col in range(1, n_cols):
            id = coord2id(row, col, n_cols) 
            if cells[id].is_blank is None:
                cells[id].update_blank(texts_pos)
            if cells[id].is_blank:
                nearest_col = find_nearest_non_blank_col(cells, row, col, texts_pos, image, n_cols)
                if nearest_col == -1: continue
                elif nearest_col <= col:
                    R_pred[row, nearest_col:col] = 1
                    if verbose: print(f'Merge cells from {nearest_col} to {col} in the first row')
                else:
                    R_pred[row, col:nearest_col] = 1
                    if verbose: print(f'Merge cells from {col} to {nearest_col} in the first row')

#######################################################################################
### RULE 3: Merge adjacent columns when the vast majority of paired cells           ###
###         (after row 3) are either both blank or only one cell per pair           ###
###         is non-blank. This merge a content column with a (mostly) blank column  ###
#######################################################################################

'''
Notes there is some modification. I compared pairs of cells starting from row 1 (not 3)
as it worked better empirically.
'''
                
def satisfies_rule_3(cells, col, texts_pos, n_rows, n_cols, th_perc, start_row):
    # checks if the column 'col' and its right column have a majority of blank cells
    cnt = 0
    total = n_rows - start_row
    for row in range(start_row, n_rows):
        id_l, id_r = coord2id(row, col, n_cols), coord2id(row, col+1, n_cols)
        
        # update cells' blank info if necessary
        if cells[id_l].is_blank is None:
            cells[id_l].update_blank(texts_pos)
        if cells[id_r].is_blank is None:
            cells[id_r].update_blank(texts_pos)
        
        # check rule 3 condition
        if (cells[id_l].is_blank and cells[id_r].is_blank) or \
           (not cells[id_l].is_blank and cells[id_r].is_blank) \
        :
            cnt += 1
    return (cnt / total) >= th_perc

def rule3(cells, texts_pos, R_pred, th_perc=0.66, verbose=False, start_row=1):
    # this function handles over-splitting problem of Split module
    if R_pred is None: 
        return
    n_rows = R_pred.shape[0]
    n_cols = R_pred.shape[1] + 1
    if n_rows <= start_row: 
        return

    for col in range(n_cols-1):
        if satisfies_rule_3(cells, col, texts_pos, n_rows, n_cols, th_perc, start_row):
            R_pred[1:, col] = 1 # rule 2 takes care of header row so this is skipped
            if verbose: print(f'Merge column {col} with column {col+1} (indexing starts at 0)')

#######################################################################################
### RULE 4: Split columns that have a consistent whitespace gap between             ###
###         vertically aligned texts                                                ###
#######################################################################################
            
# TODO
            
#######################################################################################
### RULE 5 (INVENTED): For the first column, merge blank cells                      ###
###                    with the nearest non-blank cells such that there is no       ###
###                    seperator between them                                       ###
#######################################################################################
            
def invert_threshold(img):
    '''Threhold an image and then invert values'''
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    th = 128
    _, img_th = cv.threshold(img_gray, th, 255, cv.THRESH_BINARY)
    img_inv = cv.bitwise_not(img_th) / 255.
    return img_inv
            
def has_row_sep_between(row1, row2, col, cells, image, n_cols):
    img = invert_threshold(image)
    cell1 = cells[coord2id(row1, col, n_cols)]
    cell2 = cells[coord2id(row2, col, n_cols)]
    assert cell1.left == cell2.left and cell1.right == cell2.right, 'The two cells should have the same left and right coordinates'

    if cell1.bottom < cell2.top:
        area = img[cell1.bottom : cell2.top, cell1.left : cell1.right]
    elif cell2.bottom < cell1.top:
        area = img[cell2.bottom : cell1.top, cell1.left : cell1.right]
    else:
        print('The two cells overlap')
        return None
    has_sep = (area.mean(axis=1).max() == 1.0)
    return has_sep
            
def find_nearest_non_blank_row(cells, cur_row, col, texts_pos, image, n_rows, n_cols):
    # update blank info of cells in the column if necessary
    for row in range(0, n_rows):
        id = coord2id(row, col, n_cols) # id of cells in the first column
        if cells[id].is_blank is None:
            cells[id].update_blank(texts_pos)

    # search for the nearest non-blank cell in the first column, skip the top cell
    nearest_row = -1
    best_dist = 100000
    for row in range(1, n_rows):
        if cells[coord2id(row, col, n_cols)].is_blank or has_row_sep_between(row, cur_row, col, cells, image, n_cols):
            continue
        dist = abs(cur_row - row)
        if dist < best_dist:
            nearest_row = row
            best_dist = dist
    return nearest_row

def rule5(cells, texts_pos, image, D_pred, only_first_col=True, verbose=False):
    if D_pred is None: return
    n_rows, n_cols = D_pred.shape[0] + 1, D_pred.shape[1]

    if only_first_col:
        cols = [0]
    else:
        cols = range(n_cols)

    # check conditions and merge
    for col in cols:
        for row in range(1, n_rows):
            id = coord2id(row, col, n_cols) 
            if cells[id].is_blank is None:
                cells[id].update_blank(texts_pos)
            if cells[id].is_blank:
                nearest_row = find_nearest_non_blank_row(cells, row, col, texts_pos, image, n_rows, n_cols)
                if nearest_row == -1: continue
                elif nearest_row <= row:
                    D_pred[nearest_row:row, col] = 1
                    if verbose: print(f'Merge cells from {nearest_row} to {row} in the first column')
                else:
                    D_pred[row:nearest_row, col] = 1
                    if verbose: print(f'Merge cells from {row} to {nearest_row} in the first column')

#######################################################################################
### RULE 6 (INVENTED): If there are text between two blank cells,                   ###
###         then merge them together                                                ###
#######################################################################################
                
def satisfies_rule_6_right(cell1, cell2, texts_pos):
    if not cell1.is_blank or not cell2.is_blank:
        return False
    
    for _, (l, t, r, b) in texts_pos:
        if (b < cell1.top and b < cell2.top) or \
           (t > cell1.bottom and t > cell2.bottom):
            continue
        if (cell1.right <= l and r <= cell2.left) or \
           (cell2.right <= l and r <= cell1.left):
            return True
        
    return False

def satisfies_rule_6_down(cell1, cell2, texts_pos):
    if not cell1.is_blank or not cell2.is_blank:
        return False
    
    for _, (l, t, r, b) in texts_pos:
        if (r < cell1.left and r < cell2.left) or \
           (l > cell1.right and l > cell2.right):
            continue
        if (cell1.bottom <= t and b <= cell2.top) or \
           (cell2.bottom <= t and b <= cell1.top):
            return True
        
    return False

def rule6(cells, texts_pos, R_pred, D_pred, verbose=False):
    num_rows, num_cols = get_shape(R_pred, D_pred)
    for id, cell in enumerate(cells):
        # update current cell info
        if cells[id].is_blank is None: 
            cells[id].update_blank(texts_pos)
        x, y = id2coord(id, num_cols)

        # update right and down neighbors info if necessary
        rn, dn = neighbor_RD(id, num_rows, num_cols)
        if rn and cells[rn].is_blank is None:
            cells[rn].update_blank(texts_pos)
        if dn and cells[dn].is_blank is None:
            cells[dn].update_blank(texts_pos)

        # check right neighbor condition
        if rn and R_pred is not None and \
            cell.is_blank and cells[rn].is_blank and \
            satisfies_rule_6_right(cell, cells[rn], texts_pos) \
        :
            R_pred[x, y] = 1
            if verbose: print(f'Merge right at cell ({x},{y})')

        # check down neighbor condition
        if dn and D_pred is not None and \
            cell.is_blank and cells[dn].is_blank and \
            satisfies_rule_6_down(cell, cells[dn], texts_pos) \
        :
            D_pred[x, y] = 1
            if verbose: print(f'Merge down at cell ({x},{y})')

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
        if verbose: 
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