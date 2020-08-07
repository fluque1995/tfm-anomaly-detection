import keras
import cv2 as cv
import glob
import numpy as np
import os
import random

class FeaturesGenerator(keras.utils.Sequence):

    def __init__(self, from_dir, batch_size=8, nfeats=16, shuffle=True):
        
        self.from_dir = from_dir
        self.nfeats = nfeats
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # the list of classes, built in __list_all_files
        self.classes = []
        self.data = []
        
        # prepare the list
        self.__filecount = 0
        self.__read_files()
    
        
    def __len__(self):
        """ Length of the generator
        Warning: it gives the number of loop to do, not the number of files or
        frames. The result is number_of_video/batch_size. You can use it as
        `step_per_epoch` or `validation_step` for `model.fit_generator` parameters.
        """
        return self.__filecount//self.batch_size
    
    def __getitem__(self, index):
        """ Generator needed method - return a batch of `batch_size` video
        block with `self.nfeats` for each
        """
        indexes = self.data[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = [], []
        for y, feat in indexes:
            Y.append(self.classes.index(y))
            X.append(feat)
            
        return np.array(X), keras.utils.to_categorical(
            Y, num_classes=len(self.classes))
    
    def on_epoch_end(self):
        """ When epoch has finished, random shuffle images in memory """
        if self.shuffle:
            random.shuffle(self.data)
    
    def __read_files(self):
        """ List and inject features in memory """
        self.classes = glob.glob(os.path.join(self.from_dir, '*'))
        self.classes = [os.path.basename(c) for c in self.classes]
        self.__filecount = len(glob.glob(os.path.join(self.from_dir, '*/*')))
        
        i = 1
        print("Reading data, could take a while...")
        for classname in self.classes:
            files = glob.glob(os.path.join(self.from_dir, classname, '*'))
            for file in files:
                print('\rProcessing file %d/%d' % (i, self.__filecount), end='')
                i+=1
                self.__load(classname, file)
                
        if self.shuffle:
            random.shuffle(self.data)
        
        
    def __load(self, classname, file):
        frames = np.load(file)
        step = len(frames)//self.nfeats
        frames = frames[::step]
        if len(frames) >= self.nfeats:
            frames = frames[:self.nfeats]
        
        if len(frames) == self.nfeats:
            self.data.append((classname, frames))
        else:
            print('\n%s/%s has not enough features ==> %d' % (classname, file, len(frames)))
            

