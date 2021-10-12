# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch

from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import scipy.io
from model import ReidResNet

# fp16
try:
    from apex.fp16_utils import *
except ImportError:  # will be 3.x series
    print(
        "This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0"
    )
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--pretrain", action="store_true")
parser.add_argument("--gpu_ids", default="0", type=str, help="gpu_ids: e.g. 0  0,1,2  0,2")
parser.add_argument("--which_epoch", default="1", type=str, help="0,1,2,3...or last")
parser.add_argument(
    "--test_dir",
    default="datasets/market1501/Market-1501-v15.09.15/pytorch",
    type=str,
    help="./test_data",
)
# parser.add_argument("--name", default="ft_ResNet50", type=str, help="save model path")
parser.add_argument("--batchsize", default=256, type=int, help="batchsize")
# parser.add_argument("--use_dense", action="store_true", help="use densenet121")
# parser.add_argument("--PCB", action="store_true", help="use PCB")
parser.add_argument("--multi", action="store_true", help="use multiple query")
parser.add_argument("--fp16", action="store_true", help="use fp16.")
parser.add_argument("--ibn", action="store_true", help="use ibn.")
parser.add_argument("--ms", default="1", type=str, help="multiple_scale: e.g. 1 1,1.1  1,1.1,1.2")

opt = parser.parse_args()
test_dir = opt.test_dir
#
h, w = 256, 128

data_transforms = transforms.Compose(
    [
        transforms.Resize((h, w)),
        transforms.Pad(padding=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

data_dir = test_dir

if opt.multi:
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
        for x in ["gallery", "query", "multi-query"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=16
        )
        for x in ["gallery", "query", "multi-query"]
    }
else:
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms)
        for x in ["gallery", "query"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=16
        )
        for x in ["gallery", "query"]
    }
class_names = image_datasets["query"].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
# ---------------------------
def load_network(network):
    save_path = os.path.join("./model", "pretrain_%s.pt" % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n, 2048).zero_().cuda()
        input_img = Variable(img.cuda())

        model(input_img)
        outputs = model.hooks.cuda()

        ff += outputs
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)

    return features


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split("c")[1]
        if label[0:2] == "-1":
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


gallery_path = image_datasets["gallery"].imgs
query_path = image_datasets["query"].imgs

gallery_cam, gallery_label = get_id(gallery_path)
query_cam, query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets["multi-query"].imgs
    mquery_cam, mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print("-------test-----------")
if opt.pretrain:
    num_classes = 1501
else:
    num_classes = 500

model_structure = ReidResNet(num_classes=num_classes)
model = load_network(model_structure)

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model, dataloaders["gallery"])
    query_feature = extract_feature(model, dataloaders["query"])
    if opt.multi:
        mquery_feature = extract_feature(model, dataloaders["multi-query"])

# Save to Matlab for check
result = {
    "gallery_f": gallery_feature.numpy(),
    "gallery_label": gallery_label,
    "gallery_cam": gallery_cam,
    "query_f": query_feature.numpy(),
    "query_label": query_label,
    "query_cam": query_cam,
}
scipy.io.savemat("pytorch_result.mat", result)

result = "result.txt"
os.system("python evaluate_gpu.py | tee -a %s" % result)

if opt.multi:
    result = {
        "mquery_f": mquery_feature.numpy(),
        "mquery_label": mquery_label,
        "mquery_cam": mquery_cam,
    }
    scipy.io.savemat("multi_query.mat", result)
