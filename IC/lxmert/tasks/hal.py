# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.hal_model import HalModel
from tasks.hal_data import HalDataset, HalTorchDataset

DataTuple = collections.namedtuple("DataTuple", 'dataset loader')


def get_data_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = HalDataset(splits)
    tset = HalTorchDataset(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader)


class Hal:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = HalModel()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        
        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.loss_fct = nn.CrossEntropyLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            num_correct = 0
            num_total = 0
            for i, (feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                # assert logit.dim() == target.dim() == 2
                loss = self.loss_fct(logit, target)
                # loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                pred = torch.argmax(logit, -1)
                num_correct += torch.sum(pred==target)
                num_total += pred.size()[0]

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, (num_correct/num_total) * 100.)
            # print(log_str)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def evaluate(self, eval_tuple: DataTuple):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        _, loader = eval_tuple
        pred = []
        gt = []
        for datum_tuple in loader:
            feats, boxes, sent, target = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                pred.append(torch.argmax(logit, -1))
                gt.append(target.cuda())
        pred = torch.cat(pred, 0)
        gt  = torch.cat(gt, 0)
        acc = torch.sum(pred == gt)/pred.size()[0]
        return acc

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    hal = Hal()

    # Load Hal model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        hal.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test:
            hal.evaluate(
                get_data_tuple(args.test, bs=128,
                               shuffle=False, drop_last=False)
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = hal.evaluate(
                get_data_tuple('val', bs=128,
                               shuffle=False, drop_last=False)
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', hal.train_tuple.dataset.splits)
        if hal.valid_tuple is not None:
            print('Splits in Valid data:', hal.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        hal.train(hal.train_tuple, hal.valid_tuple)


