import os
import numpy as np
import os.path as osp
import cv2
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torchvision
from dataset import BSDS_Dataset
from models import RCF
from utils import Logger, Averagvalue, Cross_entropy_loss


def train(args, model, train_loader, optimizer, epoch, logger):
    batch_time = Averagvalue()
    losses = Averagvalue()
    model.train()
    end = time.time()
    counter = 0
    for i, (image, label) in enumerate(train_loader):
        image, label = image.cuda(), label.cuda()
        outputs = model(image)
        loss = torch.zeros(1).cuda()
        for o in outputs:
            loss = loss + Cross_entropy_loss(o, label)
        counter += 1
        loss = loss / args.iter_size
        loss.backward()
        if counter == args.iter_size:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0
        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        batch_time.update(time.time() - end)
        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}/{1}][{2}/{3}] '.format(epoch + 1, args.max_epoch, i, len(train_loader)) + \
                        'Time {batch_time.val:.3f} (avg: {batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                        'Loss {loss.val:f} (avg: {loss.avg:f}) '.format(loss=losses))
        end = time.time()


def single_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    for idx, image in enumerate(test_loader):
        image = image.cuda()
        _, _, H, W = image.shape
        results = model(image)
        all_res = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
          all_res[i, 0, :, :] = results[i]
        filename = osp.splitext(test_list[idx])[0]
        torchvision.utils.save_image(1 - all_res, osp.join(save_dir, '%s.jpg' % filename))
        fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
        fuse_res = ((1 - fuse_res) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ss.png' % filename), fuse_res)
        #print('\rRunning single-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    logger.info('Running single-scale test done')


def multi_scale_test(model, test_loader, test_list, save_dir):
    model.eval()
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for idx, image in enumerate(test_loader):
        in_ = image[0].numpy().transpose((1, 2, 0))
        _, _, H, W = image.shape
        ms_fuse = np.zeros((H, W), np.float32)
        for k in range(len(scale)):
            im_ = cv2.resize(in_, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2, 0, 1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            fuse_res = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse_res = cv2.resize(fuse_res, (W, H), interpolation=cv2.INTER_LINEAR)
            ms_fuse += fuse_res
        ms_fuse = ms_fuse / len(scale)
        ### rescale trick
        # ms_fuse = (ms_fuse - ms_fuse.min()) / (ms_fuse.max() - ms_fuse.min())
        filename = osp.splitext(test_list[idx])[0]
        ms_fuse = ((1 - ms_fuse) * 255).astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, '%s_ms.png' % filename), ms_fuse)
        #print('\rRunning multi-scale test [%d/%d]' % (idx + 1, len(test_loader)), end='')
    logger.info('Running multi-scale test done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-6, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=2e-4, type=float, help='weight decay')
    parser.add_argument('--stepsize', default=3, type=int, help='learning rate step size')
    parser.add_argument('--gamma', default=0.1, type=float, help='learning rate decay rate')
    parser.add_argument('--max-epoch', default=10, type=int, help='the number of training epochs')
    parser.add_argument('--iter-size', default=10, type=int, help='iter size')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--print-freq', default=200, type=int, help='print frequency')
    parser.add_argument('--gpu', default='0', type=str, help='GPU ID')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    parser.add_argument('--save-dir', help='output folder', default='results/RCF')
    parser.add_argument('--dataset', help='root folder of dataset', default='data')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if not osp.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    logger = Logger(osp.join(args.save_dir, 'log.txt'))
    logger.info('Called with args:')
    for (key, value) in vars(args).items():
        logger.info('{0:15} | {1}'.format(key, value))

    train_dataset = BSDS_Dataset(root=args.dataset, split='train')
    test_dataset  = BSDS_Dataset(root=osp.join(args.dataset, 'HED-BSDS'), split='test')
    train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, drop_last=True, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False, shuffle=False)
    test_list = [osp.split(i.rstrip())[1] for i in test_dataset.file_list]
    assert len(test_list) == len(test_loader)

    model = RCF(pretrained='vgg16convs.mat').cuda()
    parameters = {'conv1-4.weight': [], 'conv1-4.bias': [], 'conv5.weight': [], 'conv5.bias': [],
        'conv_down_1-5.weight': [], 'conv_down_1-5.bias': [], 'score_dsn_1-5.weight': [],
        'score_dsn_1-5.bias': [], 'score_fuse.weight': [], 'score_fuse.bias': []}
    for pname, p in model.named_parameters():
        if pname in ['conv1_1.weight','conv1_2.weight',
                     'conv2_1.weight','conv2_2.weight',
                     'conv3_1.weight','conv3_2.weight','conv3_3.weight',
                     'conv4_1.weight','conv4_2.weight','conv4_3.weight']:
            parameters['conv1-4.weight'].append(p)
        elif pname in ['conv1_1.bias','conv1_2.bias',
                       'conv2_1.bias','conv2_2.bias',
                       'conv3_1.bias','conv3_2.bias','conv3_3.bias',
                       'conv4_1.bias','conv4_2.bias','conv4_3.bias']:
            parameters['conv1-4.bias'].append(p)
        elif pname in ['conv5_1.weight','conv5_2.weight','conv5_3.weight']:
            parameters['conv5.weight'].append(p)
        elif pname in ['conv5_1.bias','conv5_2.bias','conv5_3.bias']:
            parameters['conv5.bias'].append(p)
        elif pname in ['conv1_1_down.weight','conv1_2_down.weight',
                       'conv2_1_down.weight','conv2_2_down.weight',
                       'conv3_1_down.weight','conv3_2_down.weight','conv3_3_down.weight',
                       'conv4_1_down.weight','conv4_2_down.weight','conv4_3_down.weight',
                       'conv5_1_down.weight','conv5_2_down.weight','conv5_3_down.weight']:
            parameters['conv_down_1-5.weight'].append(p)
        elif pname in ['conv1_1_down.bias','conv1_2_down.bias',
                       'conv2_1_down.bias','conv2_2_down.bias',
                       'conv3_1_down.bias','conv3_2_down.bias','conv3_3_down.bias',
                       'conv4_1_down.bias','conv4_2_down.bias','conv4_3_down.bias',
                       'conv5_1_down.bias','conv5_2_down.bias','conv5_3_down.bias']:
            parameters['conv_down_1-5.bias'].append(p)
        elif pname in ['score_dsn1.weight','score_dsn2.weight','score_dsn3.weight', 'score_dsn4.weight','score_dsn5.weight']:
            parameters['score_dsn_1-5.weight'].append(p)
        elif pname in ['score_dsn1.bias','score_dsn2.bias','score_dsn3.bias', 'score_dsn4.bias','score_dsn5.bias']:
            parameters['score_dsn_1-5.bias'].append(p)
        elif pname in ['score_fuse.weight']:
            parameters['score_fuse.weight'].append(p)
        elif pname in ['score_fuse.bias']:
            parameters['score_fuse.bias'].append(p)

    optimizer = torch.optim.SGD([
            {'params': parameters['conv1-4.weight'],       'lr': args.lr*1,     'weight_decay': args.weight_decay},
            {'params': parameters['conv1-4.bias'],         'lr': args.lr*2,     'weight_decay': 0.},
            {'params': parameters['conv5.weight'],         'lr': args.lr*100,   'weight_decay': args.weight_decay},
            {'params': parameters['conv5.bias'],           'lr': args.lr*200,   'weight_decay': 0.},
            {'params': parameters['conv_down_1-5.weight'], 'lr': args.lr*0.1,   'weight_decay': args.weight_decay},
            {'params': parameters['conv_down_1-5.bias'],   'lr': args.lr*0.2,   'weight_decay': 0.},
            {'params': parameters['score_dsn_1-5.weight'], 'lr': args.lr*0.01,  'weight_decay': args.weight_decay},
            {'params': parameters['score_dsn_1-5.bias'],   'lr': args.lr*0.02,  'weight_decay': 0.},
            {'params': parameters['score_fuse.weight'],    'lr': args.lr*0.001, 'weight_decay': args.weight_decay},
            {'params': parameters['score_fuse.bias'],      'lr': args.lr*0.002, 'weight_decay': 0.},
        ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.resume is not None:
        if osp.isfile(args.resume):
            logger.info("=> loading checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            logger.info("=> checkpoint loaded")
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.max_epoch):
        logger.info('Performing initial testing...')
        train(args, model, train_loader, optimizer, epoch, logger)
        save_dir = osp.join(args.save_dir, 'epoch%d-test' % (epoch + 1))
        single_scale_test(model, test_loader, test_list, save_dir)
        multi_scale_test(model, test_loader, test_list, save_dir)
        # Save checkpoint
        save_file = osp.join(args.save_dir, 'checkpoint_epoch{}.pth'.format(epoch + 1))
        torch.save({
                'epoch': epoch,
                'args': args,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, save_file)
        lr_scheduler.step() # will adjust learning rate

    logger.close()
