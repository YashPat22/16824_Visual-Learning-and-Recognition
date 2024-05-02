from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset
import tqdm


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch+1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch+1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = 'checkpoint-{}-epoch{}.pth'.format(
        model_name, epoch+1)
    print("saving model at ", filename)
    torch.save(model, filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size)
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)

    # Ensure model is in correct mode and on right device
    model.train()
    model = model.to(args.device)

    cnt = 0

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            ##################################################################
            # TODO: Implement a suitable loss function for multi-label
            # classification. You are NOT allowed to use any pytorch built-in
            # functions. Remember to take care of underflows / overflows.
            # Function Inputs:
            #   - `output`: Outputs from the network
            #   - `target`: Ground truth labels, refer to voc_dataset.py
            #   - `wgt`: Weights (difficult or not), refer to voc_dataset.py
            # Function Outputs:
            #   - `output`: Computed loss, a single floating point number
            ##################################################################
            # log_prob = torch.nn.functional.log_softmax(output,dim=1)
            
            # output = output.detach().numpy()
            # target = target.detach().numpy()
            # wgt = wgt.detach().numpy()

            # print("Output Shape ",output.shape)
            # print("Target Shape ",target.shape)
            # print("Weight Shape ",wgt.shape)
            # loss = -np.sum(target * np.log(output), axis=1)
            # loss = -torch.sum(target * torch.log(output), dim=1)
            # print("output", output)
            # print("log out", torch.log(output))
            # print("Loss before mean ", loss)
            # loss = - torch.sum(log_prob * target) / torch.sum(target)
            # loss = torch.nn.CrossEntropyLoss(output,target)
            # loss *= wgt
            # loss = torch.mean(loss)
            # print("Loss after mean ", loss)
            # output = torch.sigmoid(output)

            # loss = -(target * torch.log(output) + (1 - target) * torch.log(1 - output))
            # loss *= wgt

            # loss = torch.sum(loss)/torch.sum(wgt)

            num = torch.exp(output- torch.max(output,dim=1,keepdim=True)[0])

            softmax = num / torch.sum(num, dim=1,keepdim=True)

            eps = 1e-8
            prob = torch.log(softmax+eps)

            loss = (-target * prob) - ((1-target)* torch.log(1 - softmax + eps))

            loss = torch.unsqueeze(wgt,1) * loss

            loss = torch.mean(loss)
            # loss = -wgt * target * torch.log(output)
            # loss = torch.mean(loss)

            # loss = torch.mean(((target-output)**2)* wgt) #MSE Loss

            # # loss = torch.from_numpy(loss)
            # loss = torch.mean(loss)

            # loss = np.mean(loss)
            # loss1 = torch.nn.CrossEntropyLoss(weight=wgt)

            # loss = loss1(output,target)



            ##################################################################
            #                          END OF YOUR CODE                      #
            ##################################################################
            
            loss.backward()
            
            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, cnt, 100. * batch_idx / len(train_loader), loss.item()))
                
                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()
            
            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()
            
            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # save model
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Validation iteration
    test_loader = utils.get_data_loader('voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size)
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    return ap, map
