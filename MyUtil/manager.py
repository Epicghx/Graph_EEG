import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from . import Metric, classification_accuracy
from MyUtil.Regularization import Regularization
import ipdb

class Manager(object):
    """Handles training and pruning."""

    def __init__(self,args, model, train_loader, test_loader):

        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader   = test_loader
        self.alpha = 1e-3

        self.device = torch.device('cuda:0')
        self.criterion = nn.MSELoss()
        self.regu = Regularization(model, args.weight_decay, 2).cuda()    # weight_decay = 4e-5
        # self.criterion = nn.KLDivLoss(reduce=True, reduction='mean')
        self.crt1 = nn.CrossEntropyLoss(reduce=True, reduction='sum')
        self.crt2 = nn.MSELoss(reduction='mean')
        return

    def kl_categorical(p_logit, q_logit):
        p = F.softmax(p_logit, dim=1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=1)
                             - F.log_softmax(q_logit, dim=1)), 1)
        return torch.mean(_kl)


    def train(self, optimizer, epoch_idx):
        # Set model to training mode
        self.model.train()

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')
        with tqdm(total=len(self.train_loader),
                  desc='Train Ep. #{}: '.format(epoch_idx + 1),
                  disable=False,
                  ascii=True) as t:
            for batch_idx, data in enumerate(self.train_loader, 0):

                inputs, labels, labelsprob = data.to(self.device), data.y.to(self.device), data.yprob.to(self.device)
                num = torch.tensor(len(data))

                optimizer.zero_grad()
                # Do forward-backward.
                # forward + backward + optimize
                # ipdb.set_trace()
                outputs, domain_output = self.model(inputs)
                labels = labels  # + torch.full( (labels.size(0), labels.size(1)),1e-10 )

                # loss = criterion(F.log_softmax(labels,dim=0), outputs)

                # loss = F.cross_entropy(outputs, labels.T[0,:].T.long())#crt2(outputs, labels) + 1e-7*RGNN.edge_weight.norm(1)

                KL_loss = F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(labelsprob, dim=1),
                                reduction='batchmean')  # good one
                # loss = F.kl_div(F.softmax(outputs,dim=1), F.log_softmax(labelsprob,dim=1))
                # loss = kl_categorical(labelsprob, outputs)

                # loss = crt2(outputs, labels)
                loss = KL_loss + self.alpha * self.model.edge_weight.norm(1) + self.regu(self.model)  ## fi' + fiD

                loss.backward()
                optimizer.step()
                train_loss.update(loss, num)
                train_accuracy.update(classification_accuracy(outputs, labels), num)
                # print(outputs)
                # print(loss)

                # print((labels.argmax(dim=1)==outputs.argmax(dim=1)).sum().numpy()/outputs.size(0))

                # lossall.append( loss.item() )
                # rateall.append((labels.argmax(dim=1)==outputs.argmax(dim=1)).sum().numpy()/outputs.size(0))

                # lossepoch = lossepoch+loss
                # rateepoch = rateepoch+(labels.argmax(dim=1)==outputs.argmax(dim=1)).sum().numpy()/outputs.size(0)


                # rateall.append((labels.T[0,:].T.long()==outputs.argmax(dim=1)).sum().float()/outputs.size(0))


                # rateepoch = rateepoch + (
                #             (labels.T[0, :].T.long() == outputs.argmax(dim=1)).sum().float() / outputs.size(0))

                # print(loss)
                # print((labels.T[0,:].T.long()==outputs.argmax(dim=1)).sum().float()/g_batch_size)

                t.set_postfix({'loss': train_loss.avg.item(),
                               'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item())
                               })
                t.update(1)

    #{{{ Evaluate classification
    def eval(self, epoch_idx, biases=None):
        """Performs evaluation."""

        self.model.eval()

        test_loss = Metric('test_loss')
        test_accuracy = Metric('test_accuracy')

        with tqdm(total=len(self.test_loader),
                  desc='Val Ep. #{}: '.format(epoch_idx + 1),
                  ascii=True) as t:
            with torch.no_grad():
                for batch_idx, data in enumerate(self.test_loader):
                    inputs, labels = data.to(self.device), data.y.to(self.device)
                    num = torch.tensor(len(data))

                    # Do forward-backward.
                    # forward + backward + optimize

                    output, domain_output = self.model(inputs)
                    # ipdb.set_trace()
                    test_loss.update(self.criterion(output, labels), num)
                    test_accuracy.update(classification_accuracy(output, labels), num)

                    t.set_postfix({
                                    'loss': test_loss.avg.item(),
                                    'accuracy': '{:.2f}'.format(100. * test_accuracy.avg.item())
                                   })
                    t.update(1)


    def save_checkpoint(self, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, arch=self.configs["arch"][self.index])
        checkpoint = {
            'model_state_dict': self.model.module.state_dict()
        }
        torch.save(checkpoint, filepath)
        return
