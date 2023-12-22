import os
import torch
import torch.backends.cudnn as cudnn

from split import train
from modules.split_modules import SplitModel

LOAD_PRETRAINED = False
MODEL_WEIGHT = '/home/hoanghuuson/table_recognition/scitsr-split-train/result/resultCP_v2.pth'

train_img_dir = '/home/hoanghuuson/table_recognition/scitsr-split-train-small/train/img'
train_json_label = '/home/hoanghuuson/table_recognition/scitsr-split-train-small/train/label/split-label.json'
val_img_dir = '/home/hoanghuuson/table_recognition/scitsr-split-train-small/val/img'
val_json_label = '/home/hoanghuuson/table_recognition/scitsr-split-train-small/val/label/split-label.json'
save_dir = '/home/hoanghuuson/table_recognition/scitsr-split-train-small/result'
pred_dir = '/home/hoanghuuson/table_recognition/scitsr-split-train-small/imgs'

class Args:
    def __init__(self, train_img_dir, train_json_label, val_img_dir, val_json_label, save_dir,
                 batch_size=1, epochs=50, gpu=True, gpu_list='0', lr=0.00075,
                 featureW=8, scale=0.5):
        self.img_dir = train_img_dir
        self.json_dir = train_json_label
        self.saved_dir = save_dir
        self.val_img_dir = val_img_dir
        self.val_json = val_json_label
        self.batch_size = batch_size
        self.epochs = epochs
        self.gpu = gpu
        self.gpu_list = gpu_list
        self.lr = lr
        self.featureW = featureW
        self.scale = scale

opt = Args(train_img_dir, train_json_label, val_img_dir, val_json_label, save_dir)
net = SplitModel(3)
net = torch.nn.DataParallel(net)
if LOAD_PRETRAINED:
    net.load_state_dict(torch.load(MODEL_WEIGHT))

print(f"Training on {'CUDA' if torch.cuda.is_available() else 'CPU'}")

if opt.gpu:
    cudnn.benchmark = True
    cudnn.deterministic = True
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_list
    net = net.cuda()

if not os.path.exists(opt.saved_dir):
    os.mkdir(opt.saved_dir)

train.train(opt, net, pred_dir)