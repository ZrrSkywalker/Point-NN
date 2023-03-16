import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import argparse

from datasets.data_seg import PartNormalDataset
from utils import *
from models import Point_NN_Seg



def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='shapenetpart')  # 71.27, 73.95

    parser.add_argument('--bz', type=int, default=1)  # Freeze as 1

    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--dim', type=int, default=144)
    parser.add_argument('--k', type=int, default=90)
    parser.add_argument('--de_k', type=int, default=6)  # propagate neighbors in decoder
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--beta', type=int, default=100)

    parser.add_argument('--gamma', type=int, default=300)  # Best as 300

    args = parser.parse_args()
    return args


@torch.no_grad()
def main():

    print('==> Loading args..')
    args = get_arguments()
    print(args)


    print('==> Preparing model..')
    point_nn = Point_NN_Seg(input_points=args.points, num_stages=args.stages,
                            embed_dim=args.dim, k_neighbors=args.k, de_neighbors=args.de_k,
                            alpha=args.alpha, beta=args.beta).cuda()
    point_nn.eval()


    print('==> Preparing data..')
    train_loader = DataLoader(PartNormalDataset(npoints=args.points, split='trainval', normalize=False), 
                                num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
    test_loader = DataLoader(PartNormalDataset(npoints=args.points, split='test', normalize=False), 
                                num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)


    print('==> Constructing Point-Memory Bank..')
    num_part, num_shape = 50, 16
    # We organize point-memory bank by 16 shape labels
    feature_memory = [[] for i in range(num_shape)]
    label_memory = [[] for i in range(num_shape)]

    for points, shape_label, part_label, norm_plt in tqdm(train_loader):
    
        # pre-process
        points = points.float().cuda().permute(0, 2, 1)
        shape_label = shape_label.long().cuda().squeeze(1)
        part_label = part_label.long().cuda()

        # Pass through the Non-Parametric Encoder + Decoder
        point_features = point_nn(points)
        # All 2048 point features in a shape
        point_features = point_features.permute(0, 2, 1)  # bz, 2048, c

        # Extracting part prototypes for a shape
        feature_memory_list = []
        label_memory_list = []

        for i in range(num_part):
            # Find the point indices for the part_label within a shape
            part_mask = (part_label == i)
            if torch.sum(part_mask) == 0:
                continue
            # Extract point features for the part_label
            part_features = point_features[part_mask]
            # Obtain part prototypes by average point features for the part_label
            part_features = part_features.mean(0).unsqueeze(0)
            
            feature_memory_list.append(part_features)
            label_memory_list.append(torch.tensor(i).unsqueeze(0))
        
        # Feature Memory: store prototypes indexed by the corresponding shape_label
        feature_memory_list = torch.cat(feature_memory_list, dim=0)
        feature_memory[int(shape_label)].append(feature_memory_list)

        # Label Memory: store labels indexed by the corresponding shape_label
        label_memory_list = torch.cat(label_memory_list, dim=0)
        label_memory_list = F.one_hot(label_memory_list, num_classes=num_part)
        label_memory[int(shape_label)].append(label_memory_list)

    # Organize the point-memory bank
    for i in range(num_shape):
        # Feature Memory
        feature_memory[i] = torch.cat(feature_memory[i], dim=0)
        feature_memory[i] /= feature_memory[i].norm(dim=-1, keepdim=True)
        feature_memory[i] = feature_memory[i].permute(1, 0)
        print("Feature Memory of the " + str(i) + "-th shape is", feature_memory[i].shape)
        # Label Memory
        label_memory[i] = torch.cat(label_memory[i], dim=0).cuda().float()


    print('==> Starting Point-NN..')
    logits_list, label_list = [], []
    for points, shape_label, part_label, norm_plt in tqdm(test_loader):

        # pre-process
        points = points.float().cuda().permute(0, 2, 1)
        shape_label = shape_label.long().cuda().squeeze(1)
        part_label = part_label.long().cuda()

        # Pass through the Non-Parametric Encoder + Decoder
        point_features = point_nn(points)
        point_features = point_features.permute(0, 2, 1).squeeze(0)  # 2048, c
        point_features /= point_features.norm(dim=-1, keepdim=True)
        
        # Similarity Matching
        Sim = point_features @ feature_memory[int(shape_label)]

        # Label Integrate
        logits = (-args.gamma * (1 - Sim)).exp() @ label_memory[int(shape_label)]
  
        logits_list.append(logits.unsqueeze(0))
        label_list.append(part_label)
            
    logits_list = torch.cat(logits_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    # Compute mIoU
    iou = compute_overall_iou(logits_list, label_list)
    miou = np.mean(iou) * 100
    
    print(f"Point-NN's part segmentation mIoU: {miou:.2f}.")


if __name__ == '__main__':
    main()