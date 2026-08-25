import argparse
import datetime
import json
import random
import sys
import time
from pathlib import Path
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)

    # training Parameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # model parameters
    # - backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned', 'fourier'),
                        help="Type of positional embedding to use on top of the image features")
    # - transformer
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # loss parameters
    # - matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="SmoothL1 point coefficient in the matching cost")
    # - loss coefficients
    parser.add_argument('--ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--point_loss_coef', default=5.0, type=float)
    parser.add_argument('--eos_coef', default=0.5, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default="SHA")
    parser.add_argument('--data_path', default="./data/ShanghaiTech/PartA", type=str)
    parser.add_argument('--wheat', action='store_true', help='Wheat mono mode')
    parser.add_argument('--pea', action='store_true', help='Pea mono mode')

    # misc parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--syn_bn', default=0, type=int)
    parser.add_argument('--resolution', default=2048, type=int,
                        help='Long-side resolution used at data preparation; matching threshold = 20 * resolution/2048')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    if args.wheat and args.pea:
        sys.exit("Error: --wheat and --pea are mutually exclusive")
    if not args.wheat and not args.pea:
        sys.exit("Error: select a species with --wheat or --pea")

    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    if args.syn_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # build optimizer
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epochs)

    # build dataset(s)
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val   = build_dataset(image_set='val',   args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val   = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val   = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val   = DataLoader(dataset_val, 1, sampler=sampler_val,
                                   drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # output directory and log
    outputs_root = "./outputs"
    if utils.is_main_process:
        output_dir = os.path.join(outputs_root, args.dataset_file, args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        run_log_name = os.path.join(output_dir, 'run_log.txt')
        with open(run_log_name, "a") as log_file:
            log_file.write('Run Log %s\n' % time.strftime("%c"))
            log_file.write("{}".format(args))
            log_file.write("parameters: {}".format(n_parameters))

    # resume
    best_val, best_epoch = float('inf'), 0  # best val MAE
    match_threshold = 20.0 * args.resolution / 2048.0
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            best_val = checkpoint.get('best_mae', float('inf'))
            best_epoch = checkpoint.get('best_epoch', 0)

    # training
    print("Start training")
    start_time = time.time()
    train_loss_history: list = []
    val_f1_history:    list = []
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()

        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        loss_vals = [v for k, v in train_stats.items() if 'loss' in k and 'unscaled' not in k]
        train_loss_history.append((epoch, float(np.mean(loss_vals)) if loss_vals else 0.0))

        if utils.is_main_process:
            with open(run_log_name, "a") as log_file:
                log_file.write('\n[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()

        # save checkpoint
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
            'best_mae': best_val,
        }, output_dir / 'checkpoint.pth')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}
        if utils.is_main_process():
            with open(run_log_name, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # evaluation
        if epoch % args.eval_freq == 0 and epoch > 0:
            t1 = time.time()
            test_stats = evaluate(model, data_loader_val, device, epoch, None,
                                  match_threshold=match_threshold)
            t2 = time.time()
            mae  = test_stats['mae']
            mse  = test_stats['mse']
            f1   = test_stats.get('f1', 0.0)
            prec = test_stats.get('precision', 0.0)
            rec  = test_stats.get('recall', 0.0)
            if mae < best_val:
                best_epoch = epoch
                best_val   = mae
            val_f1_history.append((epoch, f1))
            print("\n==========================")
            print(f"\nepoch: {epoch}  mae: {mae:.2f}  mse: {mse:.2f}")
            print(f"  F1 (threshold={match_threshold:.1f}px) : {f1*100:.1f}%"
                  f"  (P={prec*100:.1f}% R={rec*100:.1f}%)")
            print(f"  Best global  : MAE {best_val:.2f}  (epoch {best_epoch})")
            print("==========================\n")
            if utils.is_main_process():
                with open(run_log_name, "a") as log_file:
                    log_file.write(
                        f"\nepoch:{epoch} mae:{mae:.2f} mse:{mse:.2f} "
                        f"f1:{f1*100:.1f}%@{match_threshold:.1f}px "
                        f"time:{t2-t1:.1f}s "
                        f"best_mae:{best_val:.2f} best_epoch:{best_epoch}\n\n")
            if mae == best_val and utils.is_main_process():
                shutil.copyfile(output_dir / 'checkpoint.pth', output_dir / 'best_checkpoint.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        print("\n========== TRAINING SUMMARY ==========")
        print(f"  Best MAE  : {best_val:.2f}  (epoch {best_epoch})")
        print("==========================================\n")

    if train_loss_history and utils.is_main_process():
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _mpac = Path(__file__).resolve().parent.parent
        mode_label = 'pea' if args.pea else 'wheat'
        vis = _mpac / 'pet_final' / f'outputs_{mode_label}_{args.resolution}' / 'vis'
        vis.mkdir(parents=True, exist_ok=True)
        curve_name = f"learningcurve_{mode_label}_{args.resolution}.png"

        fig, ax1 = plt.subplots(figsize=(11, 5))
        epochs_t, losses = zip(*train_loss_history)
        ax1.plot(epochs_t, losses, color='#4477AA', label='Train loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='#333333')
        ax1.tick_params(axis='y')
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = [], []
        if val_f1_history:
            ax2 = ax1.twinx()
            epochs_v, f1s = zip(*val_f1_history)
            ax2.plot(epochs_v, [f * 100 for f in f1s], color='#EE4444',
                     marker='o', ms=3, label='Val F1 (%)')
            ax2.set_ylabel('Val F1 (%)', color='#EE4444')
            ax2.tick_params(axis='y', labelcolor='#EE4444')
            ax2.set_ylim(0, 100)
            lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.set_title(f'PET — Learning curve ({mode_label}, res={args.resolution})')
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        fig.tight_layout()
        out_path = vis / curve_name
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Learning curve → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
