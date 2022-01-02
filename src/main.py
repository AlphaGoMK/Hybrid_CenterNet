from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model, save_model
from models.data_parallel import DataParallel
from logger import Logger
from datasets.dataset_factory import get_dataset, CTDetDataset
from trains.train_factory import train_factory
from torch.utils.data.distributed import DistributedSampler


def main(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    Dataset = get_dataset(opt.dataset, opt.task)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    print(opt)
    # ddp init
    torch.distributed.init_process_group(backend="nccl")
    local_rank = opt.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        logger = Logger(opt)
    else:
        logger = None

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    print('{} opt gpus'.format(opt.gpus))
    print('{} opt gpus[0]'.format(opt.gpus[0]))
    # opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # opt.device = torch.device('cuda')
    opt.device = device
    print(opt.device)

    print('Creating model...')
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    model.to(device)

    if True:
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)  # baseline

        ''' cam head
        assert len([param for name, param in model.named_parameters() if 'cam_head' not in name]) + len([param for param in model.cam_head.parameters()]) == len([param for param in model.parameters()])
        optimizer = torch.optim.Adam(
            [{'params': [param for name, param in model.named_parameters() if 'base' not in name]}, 
            {'params': model.base.parameters(), 'lr': opt.prm_lr[0]}], 
            lr=opt.lr)
        '''
    else:
        print('WEAK optimizer')
        assert len([param for name, param in model.named_parameters() if 'base' not in name]) + len(
            [param for param in model.base.parameters()]) == len([param for param in model.parameters()])
        optimizer = torch.optim.Adam(
            [{'params': [param for name, param in model.named_parameters() if 'base' not in name]},
             {'params': model.base.parameters(), 'lr': opt.lr * 0.1}],  # 0.1 x lr for backbone
            lr=opt.lr)

    start_epoch = 0
    #    if opt.load_model != '':
    #        model, optimizer, start_epoch = load_model(
    #            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step, local_rank)

    if opt.load_model != '':
        model = load_model(model, opt.load_model)

    Trainer = train_factory[opt.task]
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)
    trainer = Trainer(opt, model, optimizer, local_rank)
    # trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('Setting up data...')
    # dataset_val = Dataset(opt, 'val')
    dataset_val = Dataset(opt, 'val')
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        sampler=DistributedSampler(dataset_val),
        collate_fn=CTDetDataset.collate_fn if opt.iou_loss else None # 因为batch=1不需要单独的collate_fn
    )

    if opt.test:
        _, preds = trainer.val(0, val_loader)
        val_loader.dataset.run_eval(preds, opt.save_dir)
        return
    shuffle_flag = True
    if opt.debug > 0:
        shuffle_flag = False

    dataset_train = Dataset(opt, 'train')
    train_sampler = DistributedSampler(dataset_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,  # DistributedSampler(dataset_train)
        collate_fn = CTDetDataset.collate_fn if opt.iou_loss else None # 调用聚合函数
    )
    print('train loader len {}, Dist Sampler len {}'.format(len(train_loader), len(DistributedSampler(dataset_train))))

    print('Starting training...')
    best = 1e10
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        mark = epoch if opt.save_all else 'last'
        train_sampler.set_epoch(epoch)
        log_dict_train, _ = trainer.train(epoch, train_loader)
        if local_rank == 0:
            logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            if local_rank == 0:
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0 and local_rank == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds = trainer.val(epoch, val_loader)
            for k, v in log_dict_val.items():
                if local_rank == 0:
                    logger.scalar_summary('val_{}'.format(k), v, epoch)
                    logger.write('{} {:8f} | '.format(k, v))
            if log_dict_val[opt.metric] < best and local_rank == 0:
                best = log_dict_val[opt.metric]
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        elif local_rank == 0:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        if local_rank == 0:
            logger.write('\n')
        if epoch in opt.lr_step and local_rank == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

        if True:
            if epoch in opt.lr_step:
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            '''
            if epoch in opt.lr_step:    # drop base part
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                optimizer.param_groups[0]['lr'] = lr
            if epoch in opt.prm_lr_step:    # drop prm part
                lr = opt.prm_lr[opt.prm_lr_step.index(epoch)+1]
                print('Drop PRM LR to', lr)
                optimizer.param_groups[1]['lr'] = lr
            '''
        else:  # backbone x 0.1 at weak training
            if epoch in opt.lr_step:  # drop base part
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                print('Drop LR to', lr)
                optimizer.param_groups[0]['lr'] = lr
                optimizer.param_groups[1]['lr'] = lr * 0.1  # backbone

    if local_rank == 0:
        logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
