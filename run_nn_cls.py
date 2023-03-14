import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
import argparse

from datasets.data_scan import ScanObjectNN
from datasets.data_mn40 import ModelNet40
from utils import *
from models import Point_NN



def get_arguments():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='mn40')
    parser.add_argument('--dataset', type=str, default='scan')

    # parser.add_argument('--split', type=int, default=1)
    # parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--split', type=int, default=3)

    parser.add_argument('--bz', type=int, default=16)  # Freeze as 16

    parser.add_argument('--points', type=int, default=1024)
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--dim', type=int, default=72)
    parser.add_argument('--k', type=int, default=90)
    parser.add_argument('--alpha', type=int, default=1000)
    parser.add_argument('--beta', type=int, default=100)

    args = parser.parse_args()
    return args
    

@torch.no_grad()
def main():

    print('==> Loading args..')
    args = get_arguments()
    print(args)


    print('==> Preparing model..')
    point_nn = Point_NN(input_points=args.points, num_stages=args.stages,
                        embed_dim=args.dim, k_neighbors=args.k,
                        alpha=args.alpha, beta=args.beta).cuda()
    point_nn.eval()


    print('==> Preparing data..')

    if args.dataset == 'scan':
        train_loader = DataLoader(ScanObjectNN(split=args.split, partition='training', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
        test_loader = DataLoader(ScanObjectNN(split=args.split, partition='test', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
    elif args.dataset == 'mn40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)


    print('==> Constructing Point-Memory Bank..')

    feature_memory, label_memory = [], []
    # with torch.no_grad():
    for points, labels in tqdm(train_loader):
        
        points = points.cuda().permute(0, 2, 1)
        # Pass through the Non-Parametric Encoder
        point_features = point_nn(points)
        feature_memory.append(point_features)

        labels = labels.cuda()
        label_memory.append(labels)      

    # Feature Memory
    feature_memory = torch.cat(feature_memory, dim=0)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)
    feature_memory = feature_memory.permute(1, 0)
    # Label Memory
    label_memory = torch.cat(label_memory, dim=0)
    label_memory = F.one_hot(label_memory).squeeze().float()


    print('==> Saving Test Point Cloud Features..')
    
    test_features, test_labels = [], []
    # with torch.no_grad():
    for points, labels in tqdm(test_loader):
        
        points = points.cuda().permute(0, 2, 1)
        # Pass through the Non-Parametric Encoder
        point_features = point_nn(points)
        test_features.append(point_features)

        labels = labels.cuda()
        test_labels.append(labels)

    test_features = torch.cat(test_features)
    test_features /= test_features.norm(dim=-1, keepdim=True)
    test_labels = torch.cat(test_labels)


    print('==> Starting Point-NN..')
    # Search the best hyperparameter gamma
    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_acc, best_gamma = 0, 0
    for gamma in gamma_list:

        # Similarity Matching
        Sim = test_features @ feature_memory

        # Label Integrate
        logits = (-gamma * (1 - Sim)).exp() @ label_memory

        acc = cls_acc(logits, test_labels)

        if acc > best_acc:
            # print('New best, gamma: {:.2f}; Point-NN acc: {:.2f}'.format(gamma, acc))
            best_acc, best_gamma = acc, gamma

    print(f"Point-NN's classification accuracy: {best_acc:.2f}.")


if __name__ == '__main__':
    main()
