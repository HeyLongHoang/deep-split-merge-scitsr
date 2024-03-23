import copy
from tqdm import tqdm

from data_utils.utils import *
from merge.heuristics import *
from dataset.dataset import ImageDataset
from modules.split_modules import SplitModel

RIGHT = 1
DOWN = 2

class Relation:
    def __init__(self, from_text, to_text, direction, from_id=0, to_id=0, n_blanks=0):
        self.from_text = from_text
        self.to_text = to_text
        self.direction = direction
        self.from_id = from_id
        self.to_id = to_id
        self.n_blanks = n_blanks

    def equal(self, other, compare_blanks=True):
        return self.from_text == other.from_text \
               and self.to_text == other.to_text \
               and self.direction == other.direction \
               and (self.n_blanks == other.n_blanks if compare_blanks else True)

    def __repr__(self):
        direction = 'RIGHT' if self.direction == 1 else 'DOWN'
        if self.from_id == 0 and self.to_id == 0:
            return f"Relation({self.from_text}, {self.to_text}, {direction}, n_blanks={self.n_blanks})"
        else:
            return f"Relation({self.from_text}, {self.to_text}, {direction}, {self.from_id}, {self.to_id}, n_blanks={self.n_blanks})"
        

def find_right_relations(cell, cells_list, processed_cells=set(), root=None, n_steps=0):
    rels = []
    
    # Check if the cell has already been processed
    if cell in processed_cells:
        return rels
    
    processed_cells.add(cell)  # Mark the cell as processed

    right_neighbor_l = min((c.left for c in cells_list if c.left >= cell.right), default=None)
    candidates = [c for c in cells_list 
                  if (c.left == right_neighbor_l and c.top >= cell.top and c.bottom <= cell.bottom)]

    if root is None:
        root = cell 
    for cand in candidates:
        if cand.content is not None:
            rels.append(Relation(root.content, cand.content, RIGHT, n_blanks=n_steps))
        else:
            # Find the right relations of this candidate with empty content
            empty_relations = find_right_relations(cand, cells_list, processed_cells, root, n_steps+1)
            rels.extend(empty_relations)

    return rels

def find_down_relations(cell, cells_list, processed_cells=set(), root=None, n_steps=0):
    rels = []
    
    # Check if the cell has already been processed
    if cell in processed_cells:
        return rels
    
    processed_cells.add(cell)  # Mark the cell as processed

    down_neighbor_t = min((c.top for c in cells_list if c.top >= cell.bottom), default=None)
    candidates = [c for c in cells_list
                  if (c.top == down_neighbor_t and c.left >= cell.left and c.right <= cell.right)]

    if root is None:
        root = cell
    for cand in candidates:
        if cand.content is not None:
            rels.append(Relation(root.content, cand.content, DOWN, n_blanks=n_steps))
        else:
            # Find the down relations of this candidate with empty content
            empty_relations = find_down_relations(cand, cells_list, processed_cells, root, n_steps+1)
            rels.extend(empty_relations)

    return rels

def cells2relations(cells_list):
    '''
    Convert cells into relations
    Args:
        cells_list -- list of Cells containing texts information
    Returns:
        rels -- list of Relations
    '''
    rels = []
    for cell in cells_list:
        if cell.content is None:
            continue
        rels.extend(find_right_relations(cell, cells_list))
        rels.extend(find_down_relations(cell, cells_list))
    return rels

def compare_relations(rels_pred, rels_gt, compare_blanks=True):
    '''
    Count the number of correct relation predictions
    Args:
        rels_pred -- list of predicted relations
        rels_gt -- list of ground truth relations
        compare_blanks -- boolean, whether to take the number of blank cells into account
    Returns:
        cnt -- number of correct predicted relations
    '''
    dup_rels_pred = copy.deepcopy(rels_pred)
    cnt = 0
    for rgt in rels_gt:
        to_rm = None
        for i, rp in enumerate(dup_rels_pred):
            if rgt.equal(rp, compare_blanks):
                to_rm = i
                cnt += 1
                break

        if to_rm is not None:
            dup_rels_pred = dup_rels_pred[:i] + dup_rels_pred[i:]

    return cnt

def eval_relations_macro(gt, pred, cmp_blanks=True):
    '''
    Calculate precision, recall, and F1 score at macro level
    Args:
        pred -- a list of lists of Relations from prediction (for all table images where each image contains a list of Relations)
        gt -- a list of lists of Relations from ground truth
    Returns:
        res -- dictionary containing 3 keys: 'prec', 'rec', 'f1'
    '''
    tot_prec, tot_rec, cnt = 0, 0, 0
    assert len(pred) == len(gt)

    for preds, gts in zip(pred, gt):
        n_correct = compare_relations(preds, gts, cmp_blanks)
        prec = n_correct / len(preds) if len(preds) != 0 else 0
        rec = n_correct / len(gts) if len(gts) != 0 else 0
        tot_prec += prec
        tot_rec += rec
        cnt += 1

    precision = tot_prec / cnt if cnt != 0 else 0
    recall = tot_rec / cnt if cnt != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return {'prec': precision, 'rec': recall, 'f1': f1}

def eval_relations_micro(gt, pred, cmp_blanks=True):
    '''
    Calculate precision, recall, and F1 score at micro level
    Args:
        pred -- a list of lists of Relations from prediction (for all table images where each image contains a list of Relations)
        gt -- a list of lists of Relations from ground truth
    Returns:
        res -- dictionary containing 3 keys: 'prec', 'rec', 'f1'
    '''
    TP, FP, FN = 0, 0, 0

    assert len(pred) == len(gt)

    for preds, gts in zip(pred, gt):
        n_correct = compare_relations(preds, gts, cmp_blanks)
        TP += n_correct
        FP += (len(preds) - n_correct)
        FN += (len(gts) - n_correct)

    # Handle the case when both predicted and ground truth lists are empty
    if TP == 0 and FP == 0 and FN == 0:
        precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'prec': precision, 'rec': recall, 'f1': f1}

def write_list_txt(file_path, string_list):
    """
    Write a list of strings to a text file, with each string on a new line.

    Parameters:
    - file_path (str): The path to the text file.
    - string_list (list): The list of strings to be written to the file.
    """
    try:
        with open(file_path, 'w') as file:
            for string in string_list:
                file.write(str(string) + '\n')
        print(f"Successfully wrote {len(string_list)} lines to {file_path}")
    except Exception as e:
        print(f"Error writing to {file_path}: {e}")

@torch.no_grad
def process_relations(model: SplitModel, part_dir: str, error_path=None, selected_images=None):
    '''
    Args:
        model -- trained Split model
        part_dir -- partitioned directory. Could be 'train', 'val', or 'test'
        error_path -- path to json file to write file names with error while processing
    Returns:
        gt -- list of list of Relations
        pred -- list of list of Relations
    '''
    error_names = []

    label_dir = os.path.join(part_dir, 'label')
    split_labels = load_json(os.path.join(label_dir, 'split_label.json'))
    merge_labels = load_json(os.path.join(label_dir, 'merge_label.json'))
    chunk_labels = load_json(os.path.join(label_dir, 'chunk_label.json'))

    img_dir = os.path.join(part_dir, 'img')
    dataset = ImageDataset(img_dir, split_labels, 8, scale=1, min_width=10, returns_image_name=True)
    print(f'- Loaded dataset with {len(dataset)} examples')

    gt, pred = {}, {}
    for img, _, name in tqdm(dataset):
        if selected_images is not None and name not in selected_images:
            continue
        image = img.squeeze().permute(1,2,0).detach().cpu().numpy()

        # ground truth
        r_gt, c_gt, R_gt, D_gt = load_merge_gt(merge_labels, name)
        row_gt_idxs, col_gt_idxs = borders(r_gt), borders(c_gt)
        cells_gt = get_cells(row_gt_idxs, col_gt_idxs)
        cells_merged_gt = merge_cells(cells_gt, R_gt, D_gt, verbose=False)
        if cells_merged_gt is None:
            error_names.append(name)
            continue
        
        # prediction
        r_pred, c_pred = model(img.unsqueeze(0))
        r_pred, c_pred = process_split_results(r_pred, c_pred)
        r_pred, c_pred = refine_split_results(r_pred), refine_split_results(c_pred)
        row_pred_idxs, col_pred_idxs = borders(r_pred), borders(c_pred)
        cells_pred = get_cells(row_pred_idxs, col_pred_idxs)

        # merge heuristics
        texts_pos = chunk_labels[name]
        R_pred, D_pred = create_pred_matrices(row_pred_idxs, col_pred_idxs)

        rule1(cells_pred, texts_pos, R_pred, D_pred)
        rule2(cells_pred, texts_pos, image, R_pred)
        rule3(cells_pred, texts_pos, R_pred)
        rule5(cells_pred, texts_pos, image, D_pred)
        rule6(cells_pred, texts_pos, R_pred, D_pred)

        cells_merged_pred = merge_cells(cells_pred, R_pred, D_pred, verbose=False)

        # adjacency relation
        cells_merged_gt = update_cells_content(cells_merged_gt, texts_pos)
        cells_merged_pred = update_cells_content(cells_merged_pred, texts_pos)
        rels_gt = cells2relations(cells_merged_gt)
        rels_pred = cells2relations(cells_merged_pred)
        gt[name] = rels_gt
        pred[name] = rels_pred
        
    print('Finished processing relations')
    if selected_images:
        print(f'Found {len(error_names)}/{len(selected_images)} images with label error stored at {error_path}')
    else:
        print(f'Found {len(error_names)}/{len(dataset)} images with label error stored at {error_path}')
    if error_path is not None:
        write_list_txt(error_path, error_names)
    
    return gt, pred


