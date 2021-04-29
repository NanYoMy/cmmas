import SimpleITK as sitk
import numpy as np
import glob
import random
import os
import preprocessor.tools as tools
from scipy.stats import  zscore
from dirutil.helper import sort_glob
from learn2reg.sampler import Sampler
class CHallengeSampler(Sampler):
    def __init__(self,args,is_train):
        self.args=args
        self.is_train=is_train
        self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
        self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
        self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/test_target/rez/img'))
        self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/test_target/rez/lab'))
        if len(self.img_mv)!=len(self.lab_mv):
            print("error,number of image and lab not equal")
            exit(-900)
        self.num=len(self.img_mv)
        self.nb_pairs=len(self.img_fix)*len(self.img_mv)
        self.index=0
        self.len_mv=len(self.img_mv)
        self.len_fix=len(self.img_fix)