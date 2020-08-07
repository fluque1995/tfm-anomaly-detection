import keras
import cv2 as cv
import glob
import numpy as np
import os
import random

# author: Patrice Ferlet <patrice.ferlet@smile.fr>
# licence: MIT

class VideoFrameGenerator(keras.utils.Sequence):
    '''
        Video frame generator generates batch of frames from a video directory. Videos should be
        classified in classes directories. E.g: 
            videos/class1/file1.avi
            videos/class1/file2.avi
            videos/class2/file3.avi
    '''
    def __init__(self, from_dir, batch_size=8, shape=(299, 299, 3), nbframe=16,
                 shuffle=True, transform:keras.preprocessing.image.ImageDataGenerator=None
                ):
        """
        Create a Video Frame Generator with data augmentation.
        
        Usage example:
        gen = VideoFrameGenerator('./out/videos/',
            batch_size=5,
            nbframe=3,
            transform=keras.preprocessing.image.ImageDataGenerator(rotation_range=5, horizontal_flip=True))
        
        Arguments:
        - from_dir: path to the data directory where resides videos,
            videos should be splitted in directories that are name as labels
        - batch_size: number of videos to generate
        - nbframe: number of frames per video to send
        - shuffle: boolean, shuffle data at start and after each epoch
        - transform: a keras ImageGenerator configured with random transformations
            to apply on each frame. Each video will be processed with the same
            transformation at one time to not break consistence.
        """
        
        self.from_dir = from_dir
        self.nbframe = nbframe
        self.batch_size = batch_size
        self.target_shape = shape
        self.shuffle = shuffle
        self.transform = transform
        
        # the list of classes, built in __list_all_files
        self.classes = []
        self.files = []
        self.data = []
        
        # prepare the list
        self.__filecount = 0
        self.__list_all_files()
    
        
    def __len__(self):
        """ Length of the generator
        Warning: it gives the number of loop to do, not the number of files or
        frames. The result is number_of_video/batch_size. You can use it as
        `step_per_epoch` or `validation_step` for `model.fit_generator` parameters.
        """
        return self.__filecount//self.batch_size
    
    def __getitem__(self, index):
        """ Generator needed method - return a batch of `batch_size` video
        block with `self.nbframe` for each
        """
        indexes = self.data[index*self.batch_size:(index+1)*self.batch_size]
        X, Y = self.__data_aug(indexes)
        return X, Y
    
    def on_epoch_end(self):
        """ When epoch has finished, random shuffle images in memory """
        if self.shuffle:
            random.shuffle(self.data)
    
    def __list_all_files(self):
        """ List and inject images in memory """
        self.classes = glob.glob(os.path.join(self.from_dir, '*'))
        self.classes = [os.path.basename(c) for c in self.classes]
        self.__filecount = len(glob.glob(os.path.join(self.from_dir, '*/*')))
        
        i = 1
        print("Inject frames in memory, could take a while...")
        for classname in self.classes:
            files = glob.glob(os.path.join(self.from_dir, classname, '*'))
            for file in files:
                print('\rProcessing file %d/%d' % (i, self.__filecount), end='')
                i+=1
                self.__openframe(classname, file)
                
        if self.shuffle:
            random.shuffle(self.data)
        
        
    def __openframe(self, classname, file):
        """Append ORIGNALS frames in memory, transformations are made on the fly"""
        frames = []
        vid = cv.VideoCapture(file)
        while True:
            grabbed, frame = vid.read()
            if not grabbed:
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = cv.resize(frame, self.target_shape[:2])    
            frames.append(frame)
        
        step = len(frames)//self.nbframe
        frames = frames[::step]
        if len(frames) >= self.nbframe:
            frames = frames[:self.nbframe]
        
        # add frames in memory
        frames = np.array(frames, dtype=np.float32)
        frames = keras.applications.xception.preprocess_input(frames)
        if len(frames) == self.nbframe:
            self.data.append((classname, frames))
        else:
            print('\n%s/%s has not enought frames ==> %d' % (classname, file, len(frames)))
            
    def __data_aug(self, batch):
        """ Make random transformation based on ImageGenerator arguments"""
        T = None
        if self.transform:
            T = self.transform.get_random_transform(self.target_shape[:2])
        
        X, Y = [], []
        for y, images in batch:
            Y.append(self.classes.index(y)) # label
            x = []
            for img in images:
                if T:
                    x.append(self.transform.apply_transform(img, T))
                else:
                    x.append(img)
                    
            X.append(x)

        return np.array(X), keras.utils.to_categorical(Y, num_classes=len(self.classes))
