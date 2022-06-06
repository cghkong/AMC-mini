from datetime import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn

from models.mobilenet import MobileNet


from lib.utils import accuracy, AverageMeter, progress_bar
from lib.data import get_split_dataset


def train(net,epoch,train_loader,optimizer,criterion,use_cuda,val_loader):
    net.train()

    #losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #end = time.time()
    for ep in range(epoch):
        print('\nEpoch: %d' % ep)
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3))
            #losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            #end = time.time()

            '''progress_bar(batch_idx, len(train_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc5: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))'''

        #print('loss/train {}'.format(losses.avg))
        print('acc/train_top1 {}'.format(top1.avg))
        print('acc/train_top3 {}'.format(top5.avg))

    torch.save(net.state_dict(), './checkpoints/mobilenet_imagenet_0.5flops_70.5.pth')
    evaluate(net,val_loader,use_cuda)

    #torch.save(net.state_dict(), './checkpoints/mobilenet_imagenet.pth')
    #torch.save(net.state_dict(), './checkpoints/mobilenet_imagenet_0.5flops_70.5.pth')


def evaluate(net,val_loader,use_cuda):

    criterion = nn.CrossEntropyLoss()

    if use_cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net, list(range(1)))
        cudnn.benchmark = True

    # begin eval
    net.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 3))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            # timing
            #batch_time.update(time.time() - end)
            #end = time.time()

            progress_bar(batch_idx, len(val_loader), 'Loss: {:.3f} | Acc1: {:.3f}% | Acc3: {:.3f}%'
                         .format(losses.avg, top1.avg, top5.avg))

        print('Loss: {:.3f} | Acc1: {:.3f}% | Acc3: {:.3f}%'.format(losses.avg, top1.avg, top5.avg))




if __name__ =='__main__':
    val_size = 3000
    train_loader, val_loader, n_class = get_split_dataset('imagenet', 32, 8, val_size,
                                                          data_root='dataset/imagenet/',
                                                          use_real_val=False,
                                                          shuffle=False)
    print(val_loader.dataset)
    print(train_loader.dataset)
    #model = MobileNet(20, 'normal')
    model = MobileNet(20,'0.5flops')

    use_cuda = torch.cuda.is_available()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()

    model = model.cuda()
    train(model,64,train_loader,optimizer,criterion,use_cuda,val_loader)


