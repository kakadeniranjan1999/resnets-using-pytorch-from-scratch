import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
from resnets import ResNet, BaseResidualBlock
from config_reader import read_config

model_names = {
    'resnet20': [3, 3, 3],
    'resnet32': [5, 5, 5],
    'resnet44': [7, 7, 7],
    'resnet56': [9, 9, 9],
    'resnet110': [18, 18, 18],
    'resnet1202': [200, 200, 200]
}

best_prec1 = 0


def main():
    training_configs = read_config(file_path='model_configs.yml')

    global best_prec1

    # Check the save_dir exists or not
    if not os.path.exists(training_configs['save_model']['saved_model_dir']):
        os.makedirs(training_configs['save_model']['saved_model_dir'])

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    cifar_dataset = datasets.CIFAR10(root=training_configs['load_data']['save_dir'], train=True,
                                     transform=transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(32, 4),
                                         transforms.ToTensor(),
                                         normalize]), download=True)

    train_set, valid_set = torch.utils.data.random_split(cifar_dataset, [45000, 5000])

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=training_configs['train']['batch_size'],
                                               shuffle=True,
                                               num_workers=training_configs['load_data']['workers'], pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=training_configs['train']['batch_size'],
                                               shuffle=True,
                                               num_workers=training_configs['load_data']['workers'], pin_memory=True)

    resnet_model = torch.nn.DataParallel(
        ResNet(BaseResidualBlock, model_names[training_configs['train']['arch_name'].lower()]))
    resnet_model.cuda()
    cudnn.benchmark = True

    summary(resnet_model, input_size=(3, 32, 32))

    # define loss function
    loss_func = nn.CrossEntropyLoss().cuda()

    # define SGD optimizer
    sgd_optimizer = torch.optim.SGD(resnet_model.parameters(), training_configs['train']['learning_rate'],
                                    momentum=training_configs['train']['momentum'],
                                    weight_decay=training_configs['train']['weight_decay'],
                                    )

    # define learning rate scheduler
    lr_tuner = torch.optim.lr_scheduler.MultiStepLR(sgd_optimizer,
                                                    milestones=training_configs['train']['lr_scheduler_milestones'],
                                                    )

    if training_configs['train']['arch_name'] in ['resnet1202', 'resnet110']:
        for param_group in sgd_optimizer.param_groups:
            param_group['lr'] = training_configs['train']['learning_rate'] * 0.1

    if training_configs['validate']['should_validate']:
        validate(valid_loader, resnet_model, loss_func)
        return

    for epoch in range(0, training_configs['train']['epochs']):

        print('current lr {:.5e}'.format(sgd_optimizer.param_groups[0]['lr']))
        train(train_loader, resnet_model, loss_func, sgd_optimizer, epoch)
        lr_tuner.step()

        # validate the current training progress
        prec1 = validate(valid_loader, resnet_model, loss_func)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % training_configs['save_model']['save_checkpoint_epoch'] == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': resnet_model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(training_configs['save_model']['saved_model_dir'], 'checkpoint.th'))

        save_checkpoint({
            'state_dict': resnet_model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(training_configs['save_model']['saved_model_dir'], 'model.th'))


def train(train_loader, model, loss_func, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target
        input_var = input
        target_var = target

        # forward propagation
        output = model(input_var)
        loss = loss_func(output, target_var)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 128 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, loss_func):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target
            input_var = input
            target_var = target

            # compute output
            output = model(input_var)
            loss = loss_func(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
