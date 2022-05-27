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
        self.alpha = 1e-2

        self.device = torch.device('cuda:0')
        self.criterion = nn.CrossEntropyLoss()
        self.regu = Regularization(model, args.weight_decay, 2).cuda()    # weight_decay = 4e-5
        # self.criterion = nn.KLDivLoss(reduce=True, reduction='mean')
        self.crt1 = nn.CrossEntropyLoss(reduce=True, reduction='sum')
        self.crt2 = nn.MSELoss(reduction='mean')
        self.nll_loss = nn.NLLLoss()
        return

    def kl_categorical(p_logit, q_logit):
        p = F.softmax(p_logit, dim=1)
        _kl = torch.sum(p * (F.log_softmax(p_logit, dim=1)
                             - F.log_softmax(q_logit, dim=1)), 1)
        return torch.mean(_kl)


    def train(self, optimizer, warm_up):
        # Set model to training mode
        self.model.train()

        train_loss = Metric('train_loss')
        train_accuracy = Metric('train_accuracy')

        for batch_idx, data in enumerate(self.train_loader, 0):

            inputs, labels, labelsprob = data.to(self.device), data.y.to(self.device), data.yprob.to(self.device)
            num = torch.tensor(len(data.y))

            optimizer.zero_grad()
            # Do forward-backward.
            # forward + backward + optimize
            outputs, kld_loss, drop_rates = self.model(inputs)
            labels = labels.squeeze().long()  # + torch.full( (labels.size(0), labels.size(1)),1e-10 )

            l2_reg = None
            block_index = 0
            for layer in range(self.args.num_layer):
                l2_lay_reg = None
                if layer == 0:
                    for param in self.model.gcs[str(block_index)].parameters():
                        if l2_lay_reg is None:
                            l2_lay_reg = (param ** 2).sum()
                        else:
                            l2_lay_reg += (param ** 2).sum()
                    block_index += 1

                else:
                    for iii in range(self.args.block):
                        for param in self.model.gcs[str(block_index)].parameters():
                            if l2_lay_reg is None:
                                l2_lay_reg = (param ** 2).sum()
                            else:
                                l2_lay_reg += (param ** 2).sum()
                        block_index += 1
                # ipdb.set_trace()
                l2_lay_reg = (1 - drop_rates[layer])*l2_lay_reg

                if l2_reg is None:
                    l2_reg = l2_lay_reg
                else:
                    l2_reg += l2_lay_reg
            # ipdb.set_trace()
            nll_loss = F.nll_loss(F.log_softmax(outputs, dim=0), labels)
            tot_loss = nll_loss + warm_up * kld_loss
            main_loss = tot_loss + self.args.weight_decay * l2_reg
            main_loss.backward()
            optimizer.step()

            # KL_loss = F.kl_div(F.log_softmax(outputs, dim=1), F.softmax(labelsprob, dim=1),
            #                 reduction='sum')  # good one
            # loss = KL_loss + self.alpha * self.model.edge_weight.norm(1) + self.regu(self.model)  ## fi' + fiD
            #
            # loss.backward()
            # optimizer.step()
            train_loss.update(main_loss, num)
            train_accuracy.update(classification_accuracy(outputs, labels), num)

        return train_accuracy.avg.item()

    #{{{ Evaluate classification
    def eval(self, warm_up, biases=None):
        """Performs evaluation."""

        self.model.eval()

        test_loss = Metric('test_loss')
        test_accuracy = Metric('test_accuracy')
        best_acc = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                inputs, labels = data.to(self.device), data.y.to(self.device)
                labels = labels.squeeze().long()
                num = torch.tensor(len(data.y))
                # Do forward-backward.
                # forward + backward + optimize
                for j in range(20):
                    output, _, _ = self.model(inputs)
                # ipdb.set_trace()
                test_loss.update(self.crt1(output, labels), num)
                test_accuracy.update(classification_accuracy(output, labels), num)

        return test_accuracy.avg.item()


    def save_checkpoint(self, save_folder):
        """Saves model to file."""
        filepath = self.args.checkpoint_format.format(save_folder=save_folder, arch=self.configs["arch"][self.index])
        checkpoint = {
            'model_state_dict': self.model.module.state_dict()
        }
        torch.save(checkpoint, filepath)
        return
