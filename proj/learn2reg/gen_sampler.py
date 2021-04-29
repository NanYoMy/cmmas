import SimpleITK as sitk
import numpy as np
import glob
import random
import os
import preprocessor.tools as tools
from scipy.stats import  zscore
from dirutil.helper import sort_glob
from learn2reg.sampler import Sampler
class GenSampler(Sampler):
    def __init__(self, args, type='train_sim'):
        self.args=args
        if type=='train_sim':
            self.is_train=False
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/img'))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/lab'))
        elif type=='fusion':
            self.is_train=True
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/img'))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/lab'))
            #训练融合的代码的时候，把fuse和train_target一起放入进去
            # self.img_fix_add = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/img'))
            # self.lab_fix_add = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/lab'))

            # self.img_fix=self.img_fix+self.img_fix_add
            # self.lab_fix=self.lab_fix+self.lab_fix_add

        elif type=='test':
            self.is_train=False
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/test_target/rez/img'))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/test_target/rez/lab'))
        elif type=='validate':
            self.is_train=False
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/img'))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/lab'))
        else:
            print("not support gen sampler type")
            exit(-900)
        if len(self.img_mv)!=len(self.lab_mv):
            print("error,number of image and lab not equal")
            exit(-900)
        self.num=len(self.img_mv)
        self.nb_pairs=len(self.img_fix)*len(self.img_mv)
        self.index=0
        self.len_mv=len(self.img_mv)
        self.len_fix=len(self.img_fix)