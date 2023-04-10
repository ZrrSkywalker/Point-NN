import argparse
import datetime
import logging
import numpy as np
import os
import sklearn.metrics as metrics

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader

from datasets.data_pn_scan import ScanObjectNN
import models as models
from logger import Logger
from pointnet2_ops import pointnet2_utils
from utils import progress_bar, save_model, save_args, cal_loss, CosineLRScheduler, worker_init_fn


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoint/', help='path to save checkpoint (default: ckpt)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='Point_PN_scan', help='model name')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='point number')
    parser.add_argument('--learning_rate', default=0.002, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='decay rate')
    parser.add_argument('--seed', type=int, default=7240, help='random seed')
    parser.add_argument('--workers', default=6, type=int, help='workers')
    parser.add_argument('--optim', type=str, default="adamw", help='optimizer')
    parser.add_argument('--eps', type=float, default=0.5, help='smooth loss')

    return parser.parse_args()


def main():
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
    device = 'cuda'

    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.enabled = True
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.ckpt_dir = args.ckpt_dir + args.model + message + '-' + str(args.seed)
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.ckpt_dir, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    net = models.__dict__[args.model]()
    criterion = cal_loss
    net = net.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    num_params = 0
    for p in net.parameters():
        if p.requires_grad:
            num_params += p.numel()
    printf("===============================================")
    printf("model parameters: " + str(num_params))
    printf("===============================================")

    best_test_acc = 0.
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0
    
    save_args(args)
    logger = Logger(os.path.join(args.ckpt_dir, 'log.txt'), title="ModelNet" + args.model)
    logger.set_names(["Epoch-Num", 'Learning-Rate',
                        'Train-acc',
                        'Valid-acc'])

    printf('==> Preparing data..')
    train_loader = DataLoader(ScanObjectNN(partition='training'), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn, collate_fn=None, pin_memory=True)
    test_loader = DataLoader(ScanObjectNN(partition='test'), num_workers=args.workers,
                             batch_size=64, shuffle=False, drop_last=False, collate_fn=None, pin_memory=True)

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)

    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, eps=1e-4)

    elif args.optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, eps=1e-4)

    scheduler = CosineLRScheduler(optimizer)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))

        train_out = train(net, train_loader, optimizer, criterion, args.eps, device, args.num_points)
        
        test_out = validate(net, test_loader, criterion, args.eps, device)

        scheduler.step(epoch)

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            net, epoch, path=args.ckpt_dir, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["acc"],
                       test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)

def train(net, trainloader, optimizer, criterion, eps, device, npoints=1024):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, xyz, label) in enumerate(trainloader):
        data, xyz, label = data.to(device), xyz.to(device), label.to(device).squeeze()
        data = data
        optimizer.zero_grad()
        num_curr_pts = data.shape[1]
        if num_curr_pts > npoints:
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
            if data.size(1) < point_all:
                point_all = data.size(1)
            fps_idx = pointnet2_utils.furthest_point_sample(
                data[:, :, :3].contiguous(), point_all)
            fps_idx = fps_idx[:, np.random.choice(
                point_all, npoints, False)]
            new_data = torch.gather(
                data, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, data.shape[-1]))
            new_xyz = torch.gather(
                xyz, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, xyz.shape[-1]))
        logits = net(new_data.permute(0, 2, 1), new_xyz)
        loss = criterion(logits, label, eps)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10, norm_type=2)
        optimizer.step()
        net.zero_grad()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, eps, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, xyz, label) in enumerate(testloader):
            data, xyz, label = data.to(device), xyz.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data, xyz)
            loss = criterion(logits, label, eps)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }



if __name__ == '__main__':
    main()
