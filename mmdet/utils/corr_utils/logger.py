r"""Logging"""
import datetime
import logging
import os

from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

import shutil
import time

class Logger:
    r"""Writes results of training/testing"""
    @classmethod
    def initialize(cls, args):
        logtime = datetime.datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.exp_name

        #cls.logpath = os.path.join(args.logroot, logpath + logtime + '.log')
        cls.logpath = os.path.join(args.logroot, logpath)
        cls.benchmark = args.benchmark

        if not os.path.exists(cls.logpath):
            os.makedirs(cls.logpath)

        # logging.basicConfig(filemode='w',
        #                     filename=os.path.join(cls.logpath, 'log_{}.txt'.format(logtime)),
        #                     level=logging.INFO,
        #                     format='%(message)s',
        #                     datefmt='%m-%d %H:%M:%S')
        #
        # # Console log config
        # console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(message)s')
        # console.setFormatter(formatter)
        # logging.getLogger('').addHandler(console)
        #
        def setup_logger(logger_name,log_file, level=logging.INFO):
            l = logging.getLogger(logger_name)
            formatter = logging.Formatter('%(message)s')
            fileHandler = logging.FileHandler(log_file,mode='w')
            fileHandler.setFormatter(formatter)
            streamHandler = logging.StreamHandler()
            streamHandler.setFormatter(formatter)

            l.setLevel(level)
            l.addHandler(fileHandler)
            l.addHandler(streamHandler)

        setup_logger(logger_name='log_all',log_file=os.path.join(cls.logpath, 'log_all_{}.txt'.format(args.exp_name)))
        setup_logger(logger_name='log_main', log_file=os.path.join(cls.logpath, 'log_main_{}.txt'.format(args.exp_name)))
        logger_all = logging.getLogger('log_all')
        logger_main = logging.getLogger('log_main')

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logger_all.info('\n+=========== Dynamic Hyperpixel Flow ============+')
        for arg_key in args.__dict__:
            logger_all.info('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
        logger_all.info('+================================================+\n')

        cls.logger_all = logger_all
        cls.logger_main = logger_main

    @classmethod
    def info(cls, msg):
        r"""Writes message to .txt"""
        cls.logger_all.info(msg)

    @classmethod
    def info_main(cls, msg):
        r"""Writes message to .txt for main results"""
        cls.logger_main.info(msg)

    @classmethod
    def save_model(cls, model, epoch, val_pck):
        #torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'epo{}_model.pt'.format(epoch)))
        shutil.copyfile(
            os.path.join(cls.logpath, 'epo{}_model.pt'.format(epoch)),
            os.path.join(cls.logpath, 'best_model.pt'),
        )
        cls.info('Model saved @%d w/ val. PCK: %5.2f.\n' % (epoch, val_pck))



class AverageMeter:
    r"""Stores loss, evaluation results, selected layers"""
    def __init__(self, benchamrk):
        r"""Constructor of AverageMeter"""
        if benchamrk == 'caltech':
            self.buffer_keys = ['ltacc', 'iou']
        else:
            self.buffer_keys = ['pck']

        self.buffer = {}
        for key in self.buffer_keys:
            self.buffer[key] = []

        self.loss_buffer = []

        self.time_buffer = {}

        # hsy
        # common info: cat, src_tgt img info etc
        self.common_buffer = {}
        self.common_buffer_keys = ['category', 'category_id', 'data', 'selected_pck']
        for key in self.common_buffer_keys:
            self.common_buffer[key] = []

        self.benchmark = benchamrk

    def update(self, eval_result, category, loss=None):
        for key in self.buffer_keys:
            self.buffer[key] += eval_result[key]

        if loss is not None:
            self.loss_buffer.append(loss)

    def write_result(self, split, epoch=-1):
        msg = '\n*** %s ' % split
        msg += '[@Epoch %02d] ' % epoch if epoch > -1 else ''

        if len(self.loss_buffer) > 0:
            msg += 'Loss: %5.12f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += '%s: %6.6f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))

        msg += '\tcost time {:.1f}'.format(self.get_cost_time())

        msg += '***\n'
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch=-1):
        msg = '[Student Epoch: %02d] ' % epoch if epoch > -1 else ''
        msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
        if len(self.loss_buffer) > 0:
            msg += 'Loss: %6.12f  ' % self.loss_buffer[-1]
            msg += 'Avg Loss: %6.12f  ' % (sum(self.loss_buffer) / len(self.loss_buffer))

        for key in self.buffer_keys:
            msg += 'Avg %s: %6.6f  ' % (key.upper(), sum(self.buffer[key]) / len(self.buffer[key]))
        Logger.info(msg)

    def time_start(self):
        st = time.time()
        self.time_buffer['start'] = st
        return

    def time_end(self):
        et = time.time()
        self.time_buffer['end'] = et
        return

    def get_cost_time(self):
        cost_time = self.time_buffer['end'] - self.time_buffer['start']
        return cost_time