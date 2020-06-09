# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from PIL import Image
import threading
import queue as queue # for python 3.x
import random
import numpy as np
import glob
import time
import os
import cv2
from scipy.io import loadmat

class DataFetchWorker:
    def __init__(self, factor,path, batch_size, shuffle=True):
        self.factor = factor
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
    
        self.thread_train = []
        self.thread_val = []
        self.queue_train = queue.Queue(20)
        self.queue_val = queue.Queue(20)
        self.current_idx_train = 0
        self.current_idx_val = 0
        self.stopped = True
        
        self.list_train = []
        self.list_train = glob.glob(os.path.join(self.path,'*.mat'))


        self.total_subj_train = len(self.list_train)

        if self.shuffle:
            random.shuffle(self.list_train)


    def run(self):
        self.stopped = False

        self.thread_train = threading.Thread(target=self.fill_train_batch)
        self.thread_train.setDaemon(True)
        self.thread_train.start()

    def process_data(self,coeff):
        if self.factor == 'id':
            input_coeff = coeff[0,:160].astype(np.float32)
        elif self.factor == 'exp':
            input_coeff = coeff[0,160:224].astype(np.float32)
        elif self.factor == 'rot':
            input_coeff = coeff[0,224:227].astype(np.float32)
        elif self.factor == 'gamma':
            input_coeff = coeff[0,227:254].astype(np.float32)    
        else:
            raise Exception('invalid factor')

        return input_coeff

    def get_sets(self, subjs):
        coeff_sets = []
        for subj in subjs:

        	data = loadmat(subj)
        	coeff = data['coeff']
        	coeff_ = self.process_data(coeff)
        	coeff_sets.append(coeff_)

        return coeff_sets

    def fill_train_batch(self):
        while not self.stopped:
            indices = np.array(range(self.current_idx_train, self.current_idx_train+self.batch_size)) % self.total_subj_train
            subjs = [self.list_train[subj_idx] for subj_idx in indices]
            coeff_sets = self.get_sets(subjs)
            self.queue_train.put(np.asarray(coeff_sets),\
                block=True, timeout=None)
            self.current_idx_train = (self.current_idx_train + self.batch_size) % self.total_subj_train
    

    def fetch_train_batch(self):
        coeff_sets = self.queue_train.get(block=True, timeout=None)
        return coeff_sets

    def stop(self):
        self.stopped = True
        
        while not self.queue_train.empty():
            self.queue_train.get(block=False)
        time.sleep(0.1)
        while not self.queue_train.empty():
            self.queue_train.get(block=False)