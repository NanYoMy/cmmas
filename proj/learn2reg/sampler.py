import SimpleITK as sitk
import numpy as np
import glob
import random
import os
import preprocessor.tools as tools
from scipy.stats import  zscore
from dirutil.helper import sort_glob,glob_cross_validation_files
from tool.parse import parse_arg_list
'''
4-fold验证
'''
# class Sampler():
#     def __init__(self,args,type):
#         self.args=args
#
#         if type=='train':
#             self.is_train=True
#             self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
#             self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
#             #所有的数据
#             self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/img'))+sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/img'))
#             self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/lab'))+sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/lab'))
#             if len(self.img_fix)>0:
#                 del self.img_fix[(args.fold-1)*4:(args.fold-1)*4+4]
#                 del self.lab_fix[(args.fold-1)*4:(args.fold-1)*4+4]
#
#
#         elif type == 'train_sim':
#             self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
#             self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/lab'))
#             self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/img'))
#             self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/lab'))
#         elif type == 'gen_fusion_train':
#             self.is_train = True
#             self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
#             self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/lab'))
#             #所有的数据
#             self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/img'))+sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/img'))
#             self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/lab'))+sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/lab'))
#             del self.img_fix[(args.fold-1)*4:(args.fold-1)*4+4]
#             del self.lab_fix[(args.fold-1)*4:(args.fold-1)*4+4]
#
#             #训练融合的代码的时候，把fuse和train_target一起放入进去
#             self.img_fix =self.img_fix + sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/img'))
#             self.lab_fix =self.lab_fix + sort_glob('{}/*.*'.format(args.dataset_dir + '/train_fuse_target/rez/lab'))
#
#         elif type == 'test':
#             self.is_train = False
#             self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
#             self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/lab'))
#             self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/test_target/rez/img'))
#             self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/test_target/rez/lab'))
#         elif type == 'validate':
#             self.is_train = False
#             self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
#             self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/lab'))
#             self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/img'))+sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/img'))
#             self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/lab'))+sort_glob('{}/*.*'.format(args.dataset_dir + '/validate_target/rez/lab'))
#
#             if len(self.img_fix)>0:
#                 self.img_fix=[self.img_fix[i] for i in range((args.fold-1)*4,(args.fold-1)*4+4)]
#                 self.lab_fix=[self.lab_fix[i] for i in range((args.fold-1)*4,(args.fold-1)*4+4)]
#
#
#         else:
#             print("not support gen sampler type")
#             exit(-900)
#
#         if len(self.img_mv)!=len(self.lab_mv):
#             print("error,number of image and lab not equal")
#             exit(-900)
#         self.num=len(self.img_mv)
#         self.nb_pairs=len(self.img_fix)*len(self.img_mv)
#         self.len_fix=len(self.img_fix)
#         self.len_mv=len(self.img_mv)
#         self.index=0
#     def reset_sequnce_index(self):
#         self.index=0
#     def next_sample(self):
#         index_mvs=[]
#         index_fixs=[]
#         for i in range(self.args.batch_size):
#             if self.is_train:
#                 index_mv,index_fix=self.generate_random_index()
#             else:
#                 index_mv,index_fix=self.generate_sequnce_index()
#             index_mvs.append(index_mv)
#             index_fixs.append(index_fix)
#             # print(str(index_mv)+":"+str(index_fix))
#         return self.get_batch_data(index_mvs,index_fixs)
#
#     def get_batch_file(self):
#         img_mvs=[]
#         img_fixs=[]
#         lab_mvs=[]
#         lab_fixs=[]
#         for i in range(self.args.batch_size):
#             if self.is_train:
#                 index_mv,index_fix=self.generate_random_index()
#             else:
#                 index_mv,index_fix=self.generate_sequnce_index()
#             img_mvs.append(self.img_mv[index_mv])
#             lab_mvs.append(self.lab_mv[index_mv])
#
#             img_fixs.append(self.img_fix[index_fix])
#             lab_fixs.append(self.lab_fix[index_fix])
#         return img_mvs,img_fixs,lab_mvs,lab_fixs
#     def get_batch_data_V2(self,img_mvs,img_fixs,lab_mvs,lab_fixs):
#         fix_imgs = []
#         fix_labs = []
#         mv_imgs = []
#         mv_labs = []
#         for img_mv,img_fix,lab_mv,lab_fix in zip(img_mvs,img_fixs,lab_mvs,lab_fixs):
#             # print(str(index_mv)+":"+str(index_fix))
#             imgA, imgB = sitk.ReadImage(img_mv), sitk.ReadImage(img_fix)
#             imgA, imgB = sitk.RescaleIntensity(imgA), sitk.RescaleIntensity(imgB)
#             labA, labB = sitk.ReadImage(lab_mv), sitk.ReadImage(lab_fix)
#             mv_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgA), axis=None), axis=-1))
#             fix_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgB), axis=None), axis=-1))
#
#             if self.is_train:
#                 # 可以选择不同的label来做evaluate
#                 candidate_label_index = [int(i) for i in self.args.components.split(',')]
#                 label_index = candidate_label_index[np.random.randint(len(candidate_label_index))]
#             else:
#                 label_index = self.args.component
#
#             '''
#             当数据是D*W*H的时候，这个数据的值可能是[1,NB_lable],所以需要进行np.where转换
#             当数据是D*W*H*C的时候，这个数据如果有通道C，则里面的数据保证为0,1
#             最后神经网络的输入数据为 D*W*H*1 ，在本网络中，支持的是这种数据
#             '''
#             labA = sitk.GetArrayFromImage(labA)
#             labA = np.where(labA == label_index, 1, 0)
#             mv_labs.append(np.expand_dims(labA, axis=-1))
#
#             labB = sitk.GetArrayFromImage(labB)
#             labB = np.where(labB == label_index, 1, 0)
#             fix_labs.append(np.expand_dims(labB, axis=-1))
#
#         fix_imgs = np.array(fix_imgs).astype(np.float32)
#         fix_labs = np.array(fix_labs).astype(np.float32)
#         mv_imgs = np.array(mv_imgs).astype(np.float32)
#         mv_labs = np.array(mv_labs).astype(np.float32)
#
#         return fix_imgs, fix_labs, mv_imgs, mv_labs
#
#     def generate_sequnce_index(self):
#         index_mv=self.index//len(self.img_fix)
#         index_fix=self.index%len(self.img_fix)
#         self.index=self.index+1
#         self.index=self.index%(len(self.img_fix)*len(self.img_mv))
#         return  index_mv,index_fix
#     def generate_random_index(self):
#         return  np.random.randint(self.num),np.random.randint(self.num)
#
#     def get_batch_data(self,atlas,targets):
#         fix_imgs = []
#         fix_labs = []
#         mv_imgs = []
#         mv_labs = []
#         for index_mv,index_fix in zip(atlas,targets):
#             # print(str(index_mv)+":"+str(index_fix))
#
#             imgA, imgB = sitk.ReadImage(self.img_mv[index_mv]), sitk.ReadImage(self.img_fix[index_fix])
#             imgA, imgB = sitk.RescaleIntensity(imgA), sitk.RescaleIntensity(imgB)
#             labA, labB = sitk.ReadImage(self.lab_mv[index_mv]), sitk.ReadImage(self.lab_fix[index_fix])
#             mv_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgA), axis=None), axis=-1))
#             fix_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgB), axis=None), axis=-1))
#             # imgA, imgB = sitk.RescaleIntensity(imgA,0,1), sitk.RescaleIntensity(imgB,0,1)
#             # labA, labB = sitk.ReadImage(self.lab_mv[index_mv]), sitk.ReadImage(self.lab_fix[index_fix])
#             # mv_imgs.append(np.expand_dims((sitk.GetArrayFromImage(imgA)), axis=-1))
#             # fix_imgs.append(np.expand_dims((sitk.GetArrayFromImage(imgB)), axis=-1))
#
#
#             if self.is_train:
#                 #可以选择不同的label来做evaluate
#                 candidate_label_index   =  [ int(i) for i in self.args.components.split(',')]
#                 label_index=candidate_label_index[np.random.randint(len(candidate_label_index))]
#             else:
#                 label_index=self.args.component
#
#             '''
#             当数据是D*W*H的时候，这个数据的值可能是[1,NB_lable],所以需要进行np.where转换
#             当数据是D*W*H*C的时候，这个数据如果有通道C，则里面的数据保证为0,1
#             最后神经网络的输入数据为 D*W*H*1 ，在本网络中，支持的是这种数据
#             '''
#             labA=sitk.GetArrayFromImage(labA)
#             labA=np.where(labA == label_index, 1, 0)
#             mv_labs.append(np.expand_dims(labA,axis=-1))
#
#             labB=sitk.GetArrayFromImage(labB)
#             labB=np.where(labB == label_index, 1, 0)
#             fix_labs.append(np.expand_dims(labB,axis=-1))
#
#         fix_imgs = np.array(fix_imgs).astype(np.float32)
#         fix_labs = np.array(fix_labs).astype(np.float32)
#         mv_imgs = np.array(mv_imgs).astype(np.float32)
#         mv_labs = np.array(mv_labs).astype(np.float32)
#
#         return fix_imgs, fix_labs,mv_imgs,mv_labs
class Sampler():
    def __init__(self,args,type):
        self.args=args

        if type=='train':
            self.is_train=True
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
            #所有的数据
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/img'))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/lab'))

            del self.img_mv[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.lab_mv[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.img_fix[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.lab_fix[(args.fold-1)*5:(args.fold-1)*5+5]
            #for similarity
            del self.img_mv[-4:]
            del self.lab_mv[-4:]
            del self.img_fix[-4:]
            del self.lab_fix[-4:]

        elif type == 'validate':
            self.is_train = False
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
            #所有的数据
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/img'))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/lab'))

            #validation
            del self.img_mv[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.lab_mv[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.img_mv[-4:]
            del self.lab_mv[-4:]

            self.img_fix = [self.img_fix[i] for i in range((args.fold - 1) * 5, (args.fold - 1) * 5 + 5)]
            self.lab_fix = [self.lab_fix[i] for i in range((args.fold - 1) * 5, (args.fold - 1) * 5 + 5)]

        elif type == 'gen_fusion_train':
            self.is_train = True
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_atlas/rez/img'))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir +'/train_atlas/rez/lab' ))
            #所有的数据
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/img'))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/train_target/rez/lab'))

            del self.img_mv[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.lab_mv[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.img_fix[(args.fold-1)*5:(args.fold-1)*5+5]
            del self.lab_fix[(args.fold-1)*5:(args.fold-1)*5+5]

        else:
            print("not support gen sampler type")
            exit(-900)

        if len(self.img_mv)!=len(self.lab_mv):
            print("error,number of image and lab not equal")
            exit(-900)
        self.num=len(self.img_mv)
        self.nb_pairs=len(self.img_fix)*len(self.img_mv)
        self.len_fix=len(self.img_fix)
        self.len_mv=len(self.img_mv)
        self.index=0
    def reset_sequnce_index(self):
        self.index=0
    def next_sample(self):
        index_mvs=[]
        index_fixs=[]
        for i in range(self.args.batch_size):
            if self.is_train:
                index_mv,index_fix=self.generate_random_index()
            else:
                index_mv,index_fix=self.generate_sequnce_index()
            index_mvs.append(index_mv)
            index_fixs.append(index_fix)
            # print(str(index_mv)+":"+str(index_fix))
        return self.get_batch_data(index_mvs,index_fixs)

    def get_batch_file(self):
        img_mvs=[]
        img_fixs=[]
        lab_mvs=[]
        lab_fixs=[]
        for i in range(self.args.batch_size):
            if self.is_train:
                index_mv,index_fix=self.generate_random_index()
            else:
                index_mv,index_fix=self.generate_sequnce_index()
            img_mvs.append(self.img_mv[index_mv])
            lab_mvs.append(self.lab_mv[index_mv])

            img_fixs.append(self.img_fix[index_fix])
            lab_fixs.append(self.lab_fix[index_fix])
        return img_mvs,img_fixs,lab_mvs,lab_fixs
    def get_batch_data_V2(self,img_mvs,img_fixs,lab_mvs,lab_fixs):
        fix_imgs = []
        fix_labs = []
        mv_imgs = []
        mv_labs = []
        for img_mv,img_fix,lab_mv,lab_fix in zip(img_mvs,img_fixs,lab_mvs,lab_fixs):
            # print(str(index_mv)+":"+str(index_fix))
            imgA, imgB = sitk.ReadImage(img_mv), sitk.ReadImage(img_fix)
            imgA, imgB = sitk.RescaleIntensity(imgA), sitk.RescaleIntensity(imgB)
            labA, labB = sitk.ReadImage(lab_mv), sitk.ReadImage(lab_fix)
            mv_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgA), axis=None), axis=-1))
            fix_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgB), axis=None), axis=-1))

            if self.is_train:
                # 可以选择不同的label来做evaluate
                candidate_label_index = [int(i) for i in self.args.components.split(',')]
                label_index = candidate_label_index[np.random.randint(len(candidate_label_index))]
            else:
                label_index = self.args.component

            '''
            当数据是D*W*H的时候，这个数据的值可能是[1,NB_lable],所以需要进行np.where转换
            当数据是D*W*H*C的时候，这个数据如果有通道C，则里面的数据保证为0,1
            最后神经网络的输入数据为 D*W*H*1 ，在本网络中，支持的是这种数据
            '''
            labA = sitk.GetArrayFromImage(labA)
            labA = np.where(labA == label_index, 1, 0)
            mv_labs.append(np.expand_dims(labA, axis=-1))

            labB = sitk.GetArrayFromImage(labB)
            labB = np.where(labB == label_index, 1, 0)
            fix_labs.append(np.expand_dims(labB, axis=-1))

        fix_imgs = np.array(fix_imgs).astype(np.float32)
        fix_labs = np.array(fix_labs).astype(np.float32)
        mv_imgs = np.array(mv_imgs).astype(np.float32)
        mv_labs = np.array(mv_labs).astype(np.float32)

        return fix_imgs, fix_labs, mv_imgs, mv_labs

    def generate_sequnce_index(self):
        index_mv=self.index//len(self.img_fix)
        index_fix=self.index%len(self.img_fix)
        self.index=self.index+1
        self.index=self.index%(len(self.img_fix)*len(self.img_mv))
        return  index_mv,index_fix
    def generate_random_index(self):
        return  np.random.randint(self.num),np.random.randint(self.num)

    def get_batch_data(self,atlas,targets):
        fix_imgs = []
        fix_labs = []
        mv_imgs = []
        mv_labs = []
        for index_mv,index_fix in zip(atlas,targets):
            # print(str(index_mv)+":"+str(index_fix))

            imgA, imgB = sitk.ReadImage(self.img_mv[index_mv]), sitk.ReadImage(self.img_fix[index_fix])
            imgA, imgB = sitk.RescaleIntensity(imgA), sitk.RescaleIntensity(imgB)
            labA, labB = sitk.ReadImage(self.lab_mv[index_mv]), sitk.ReadImage(self.lab_fix[index_fix])
            mv_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgA), axis=None), axis=-1))
            fix_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgB), axis=None), axis=-1))
            # imgA, imgB = sitk.RescaleIntensity(imgA,0,1), sitk.RescaleIntensity(imgB,0,1)
            # labA, labB = sitk.ReadImage(self.lab_mv[index_mv]), sitk.ReadImage(self.lab_fix[index_fix])
            # mv_imgs.append(np.expand_dims((sitk.GetArrayFromImage(imgA)), axis=-1))
            # fix_imgs.append(np.expand_dims((sitk.GetArrayFromImage(imgB)), axis=-1))


            if self.is_train:
                #可以选择不同的label来做evaluate
                candidate_label_index   =  [ int(i) for i in self.args.components.split(',')]
                label_index=candidate_label_index[np.random.randint(len(candidate_label_index))]
            else:
                label_index=self.args.component

            '''
            当数据是D*W*H的时候，这个数据的值可能是[1,NB_lable],所以需要进行np.where转换
            当数据是D*W*H*C的时候，这个数据如果有通道C，则里面的数据保证为0,1
            最后神经网络的输入数据为 D*W*H*1 ，在本网络中，支持的是这种数据
            '''
            labA=sitk.GetArrayFromImage(labA)
            labA=np.where(labA == label_index, 1, 0)
            mv_labs.append(np.expand_dims(labA,axis=-1))

            labB=sitk.GetArrayFromImage(labB)
            labB=np.where(labB == label_index, 1, 0)
            fix_labs.append(np.expand_dims(labB,axis=-1))

        fix_imgs = np.array(fix_imgs).astype(np.float32)
        fix_labs = np.array(fix_labs).astype(np.float32)
        mv_imgs = np.array(mv_imgs).astype(np.float32)
        mv_labs = np.array(mv_labs).astype(np.float32)

        return fix_imgs, fix_labs,mv_imgs,mv_labs
'''
用于做跨模态交叉验证的时候sampler
'''
class RegSampler():
    def __init__(self, args, type):
        self.args = args
        validation_size=5
        mode=parse_arg_list(args.mode)
        if type == 'train':
            self.is_train = True
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode[0])))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode[0])))
            # 所有的数据

            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode[1])))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode[1])))
            if len(self.img_fix) > 0:
                del self.img_fix[(args.fold - 1) * validation_size:(args.fold - 1) * validation_size + validation_size]
                del self.lab_fix[(args.fold - 1) * validation_size:(args.fold - 1) * validation_size + validation_size]
            if len(self.img_mv) > 0:
                del self.img_mv[(args.fold - 1) * validation_size:(args.fold - 1) * validation_size + validation_size]
                del self.lab_mv[(args.fold - 1) * validation_size:(args.fold - 1) * validation_size + validation_size]

        elif type == 'validate':
            self.is_train = False
            self.img_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode[0])))
            self.lab_mv = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode[0])))
            self.img_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode[1])))
            self.lab_fix = sort_glob('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode[1])))
            if len(self.img_fix)>0:
                self.img_fix=[self.img_fix[i] for i in range((args.fold-1)*validation_size,(args.fold-1)*validation_size+validation_size)]
                self.lab_fix=[self.lab_fix[i] for i in range((args.fold-1)*validation_size,(args.fold-1)*validation_size+validation_size)]
            if len(self.img_mv)>0:
                self.img_mv=[self.img_mv[i] for i in range((args.fold-1)*validation_size,(args.fold-1)*validation_size+validation_size)]
                self.lab_mv=[self.lab_mv[i] for i in range((args.fold-1)*validation_size,(args.fold-1)*validation_size+validation_size)]

        else:
            print("not support gen sampler type")
            exit(-900)

        if len(self.img_mv)!=len(self.lab_mv):
            print("error,number of image and lab not equal")
            exit(-900)
        self.num=len(self.img_mv)
        self.nb_pairs=len(self.img_fix)*len(self.img_mv)
        self.len_fix=len(self.img_fix)
        self.len_mv=len(self.img_mv)
        self.index=0
    def reset_sequnce_index(self):
        self.index=0
    def next_sample(self):
        index_mvs=[]
        index_fixs=[]
        for i in range(self.args.batch_size):
            if self.is_train:
                index_mv,index_fix=self.generate_random_index()
            else:
                index_mv,index_fix=self.generate_sequnce_index()
            index_mvs.append(index_mv)
            index_fixs.append(index_fix)
            # print(str(index_mv)+":"+str(index_fix))
        return self.get_batch_data(index_mvs,index_fixs)

    def get_batch_file(self):
        img_mvs=[]
        img_fixs=[]
        lab_mvs=[]
        lab_fixs=[]
        for i in range(self.args.batch_size):
            if self.is_train:
                index_mv,index_fix=self.generate_random_index()
            else:
                index_mv,index_fix=self.generate_sequnce_index()
            img_mvs.append(self.img_mv[index_mv])
            lab_mvs.append(self.lab_mv[index_mv])

            img_fixs.append(self.img_fix[index_fix])
            lab_fixs.append(self.lab_fix[index_fix])
        return img_mvs,img_fixs,lab_mvs,lab_fixs
    def get_batch_data_V2(self,img_mvs,img_fixs,lab_mvs,lab_fixs):
        fix_imgs = []
        fix_labs = []
        mv_imgs = []
        mv_labs = []
        for img_mv,img_fix,lab_mv,lab_fix in zip(img_mvs,img_fixs,lab_mvs,lab_fixs):
            # print(str(index_mv)+":"+str(index_fix))
            imgA, imgB = sitk.ReadImage(img_mv), sitk.ReadImage(img_fix)
            imgA, imgB = sitk.RescaleIntensity(imgA), sitk.RescaleIntensity(imgB)
            labA, labB = sitk.ReadImage(lab_mv), sitk.ReadImage(lab_fix)
            mv_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgA), axis=None), axis=-1))
            fix_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgB), axis=None), axis=-1))

            if self.is_train:
                # 可以选择不同的label来做evaluate
                candidate_label_index = [int(i) for i in self.args.components.split(',')]
                label_index = candidate_label_index[np.random.randint(len(candidate_label_index))]
            else:
                label_index = self.args.component

            '''
            当数据是D*W*H的时候，这个数据的值可能是[1,NB_lable],所以需要进行np.where转换
            当数据是D*W*H*C的时候，这个数据如果有通道C，则里面的数据保证为0,1
            最后神经网络的输入数据为 D*W*H*1 ，在本网络中，支持的是这种数据
            '''
            labA = sitk.GetArrayFromImage(labA)
            labA = np.where(labA == label_index, 1, 0)
            mv_labs.append(np.expand_dims(labA, axis=-1))

            labB = sitk.GetArrayFromImage(labB)
            labB = np.where(labB == label_index, 1, 0)
            fix_labs.append(np.expand_dims(labB, axis=-1))

        fix_imgs = np.array(fix_imgs).astype(np.float32)
        fix_labs = np.array(fix_labs).astype(np.float32)
        mv_imgs = np.array(mv_imgs).astype(np.float32)
        mv_labs = np.array(mv_labs).astype(np.float32)

        return fix_imgs, fix_labs, mv_imgs, mv_labs

    def generate_sequnce_index(self):
        index_mv=self.index//len(self.img_fix)
        index_fix=self.index%len(self.img_fix)
        self.index=self.index+1
        self.index=self.index%(len(self.img_fix)*len(self.img_mv))
        return  index_mv,index_fix
    def generate_random_index(self):
        return  np.random.randint(self.num),np.random.randint(self.num)

    def get_batch_data(self,atlas,targets):
        fix_imgs = []
        fix_labs = []
        mv_imgs = []
        mv_labs = []
        for index_mv,index_fix in zip(atlas,targets):
            # print(str(index_mv)+":"+str(index_fix))

            imgA, imgB = sitk.ReadImage(self.img_mv[index_mv]), sitk.ReadImage(self.img_fix[index_fix])
            imgA, imgB = sitk.RescaleIntensity(imgA), sitk.RescaleIntensity(imgB)
            labA, labB = sitk.ReadImage(self.lab_mv[index_mv]), sitk.ReadImage(self.lab_fix[index_fix])
            mv_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgA), axis=None), axis=-1))
            fix_imgs.append(np.expand_dims(zscore(sitk.GetArrayFromImage(imgB), axis=None), axis=-1))
            # imgA, imgB = sitk.RescaleIntensity(imgA,0,1), sitk.RescaleIntensity(imgB,0,1)
            # labA, labB = sitk.ReadImage(self.lab_mv[index_mv]), sitk.ReadImage(self.lab_fix[index_fix])
            # mv_imgs.append(np.expand_dims((sitk.GetArrayFromImage(imgA)), axis=-1))
            # fix_imgs.append(np.expand_dims((sitk.GetArrayFromImage(imgB)), axis=-1))


            # if self.is_train:
            #     可以选择不同的label来做evaluate
                # candidate_label_index   =  [ int(i) for i in self.args.components.split(',')]
                # label_index=candidate_label_index[np.random.randint(len(candidate_label_index))]
            # else:
            label_index=self.args.component

            '''
            当数据是D*W*H的时候，这个数据的值可能是[1,NB_lable],所以需要进行np.where转换
            当数据是D*W*H*C的时候，这个数据如果有通道C，则里面的数据保证为0,1
            最后神经网络的输入数据为 D*W*H*1 ，在本网络中，支持的是这种数据
            '''
            labA=sitk.GetArrayFromImage(labA)
            labA=np.where(labA == label_index, 1, 0)
            mv_labs.append(np.expand_dims(labA,axis=-1))

            labB=sitk.GetArrayFromImage(labB)
            labB=np.where(labB == label_index, 1, 0)
            fix_labs.append(np.expand_dims(labB,axis=-1))

        fix_imgs = np.array(fix_imgs).astype(np.float32)
        fix_labs = np.array(fix_labs).astype(np.float32)
        mv_imgs = np.array(mv_imgs).astype(np.float32)
        mv_labs = np.array(mv_labs).astype(np.float32)

        return fix_imgs, fix_labs,mv_imgs,mv_labs

from tool.parse import parse_arg_list
class MMSampler():
    def __init__(self, args, type):
        self.args = args
        mode_list = parse_arg_list(args.mode, 'str')
        if type == 'train':
            self.is_train = True
            self.img_mv1 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode_list[0])))
            self.lab_mv1 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode_list[0])))
            self.img_mv2 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%mode_list[1]))
            self.lab_mv2 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%mode_list[1]))
            self.img_fix = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%mode_list[2]))
            self.lab_fix = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%mode_list[2]))
            self.img_fix,_=glob_cross_validation_files(self.img_fix,5,args.fold)
            self.lab_fix,_=glob_cross_validation_files(self.lab_fix,5,args.fold)
            self.img_mv1,_=glob_cross_validation_files(self.img_mv1,5,args.fold)
            self.lab_mv1,_=glob_cross_validation_files(self.lab_mv1,5,args.fold)
            self.img_mv2,_=glob_cross_validation_files(self.img_mv2,5,args.fold)
            self.lab_mv2,_=glob_cross_validation_files(self.lab_mv2,5,args.fold)
        elif type == 'validate':
            self.is_train = False
            self.img_mv1 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode_list[0])))
            self.lab_mv1 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode_list[0])))
            self.img_mv2 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode_list[1])))
            self.lab_mv2 = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode_list[1])))
            self.img_fix = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/img'%(mode_list[2])))
            self.lab_fix = ('{}/*.*'.format(args.dataset_dir + '/%s/rez/lab'%(mode_list[2])))
            _,self.img_fix=glob_cross_validation_files(self.img_fix,5,args.fold)
            _,self.lab_fix=glob_cross_validation_files(self.lab_fix,5,args.fold)
            _,self.img_mv1=glob_cross_validation_files(self.img_mv1,5,args.fold)
            _,self.lab_mv1=glob_cross_validation_files(self.lab_mv1,5,args.fold)
            _,self.img_mv2=glob_cross_validation_files(self.img_mv2,5,args.fold)
            _,self.lab_mv2=glob_cross_validation_files(self.lab_mv2,5,args.fold)
        else:
            print("not support gen sampler type")
            exit(-900)

        if len(self.img_mv1)!=len(self.lab_mv1):
            print("error,number of image and lab not equal")
            exit(-900)
        self.num=len(self.img_mv1)
        self.nb_pairs=len(self.img_fix)*len(self.img_mv1)
        self.len_fix=len(self.img_fix)
        self.len_mv1=len(self.img_mv1)
        self.len_mv2=len(self.img_mv2)
        self.index=0
    def next_sample(self):
        mv_img1, mv_lab1, mv_img2, mv_lab2, fix_img, fix_lab= self.get_data_path()
        return self.get_batch_data(mv_img1,mv_lab1,mv_img2,mv_lab2,fix_img,fix_lab)

    def get_data_path(self ):
        mv_img1=[]
        mv_img2=[]
        mv_lab1=[]
        mv_lab2=[]
        fix_img=[]
        fix_lab=[]
        for i in range(self.args.batch_size):
            if self.is_train:
                index_mv, index_fix = self.generate_random_index()
            else:
                index_mv, index_fix = self.generate_sequnce_index()
            mv_img1.append(self.img_mv1[index_mv])
            mv_img2.append(self.img_mv2[index_mv])
            mv_lab1.append(self.lab_mv1[index_mv])
            mv_lab2.append(self.lab_mv2[index_mv])
            fix_img.append(self.img_fix[index_fix])
            fix_lab.append(self.lab_fix[index_fix])
            # print(str(index_mv)+":"+str(index_fix))
        return  mv_img1, mv_lab1, mv_img2,mv_lab2,fix_img, fix_lab


    def get_batch_data(self, mv_img1s, mv_lab1s, mv_img2s, mv_lab2s, fix_imgs, fix_labs):
        arr_mv_img1s = []
        arr_mv_lab1s = []
        arr_mv_img2s = []
        arr_mv_lab2s = []
        arr_fix_imgs = []
        arr_fix_labs = []
        for  mv_img1,mv_lab1,mv_img2,mv_lab2,fix_img,fix_lab in zip(mv_img1s, mv_lab1s, mv_img2s, mv_lab2s, fix_imgs, fix_labs):
            arr_fix_img,arr_fix_lab=self.read_data( fix_img, fix_lab)
            arr_fix_imgs.append(arr_fix_img)
            arr_fix_labs.append(arr_fix_lab)
            arr_mv1_img,arr_mv1_lab=self.read_data(  mv_img1, mv_lab1)
            arr_mv_img1s.append(arr_mv1_img)
            arr_mv_lab1s.append(arr_mv1_lab)
            arr_mv2_img,arr_mv2_lab=self.read_data(  mv_img2, mv_lab2)
            arr_mv_img2s.append(arr_mv2_img)
            arr_mv_lab2s.append(arr_mv2_lab)

        ret_mv_img1s = np.array(arr_mv_img1s).astype(np.float32)
        ret_mv_lab1s = np.array(arr_mv_lab1s).astype(np.float32)
        ret_mv_img2s = np.array(arr_mv_img2s).astype(np.float32)
        ret_mv_lab2s = np.array(arr_mv_lab2s).astype(np.float32)
        ret_fix_imgs = np.array(arr_fix_imgs).astype(np.float32)
        ret_fix_labs = np.array(arr_fix_labs).astype(np.float32)

        return ret_mv_img1s,ret_mv_lab1s,ret_mv_img2s,ret_mv_lab2s,ret_fix_imgs,ret_fix_labs

    def read_data(self, img, lab):
        # print(str(index_mv)+":"+str(index_fix))
        sitk_mv_img = sitk.ReadImage(img)
        sitk_mv_img= sitk.RescaleIntensity(sitk_mv_img)
        arr_mv_img=np.expand_dims(zscore(sitk.GetArrayFromImage(sitk_mv_img), axis=None), axis=-1)
        sitk_mv_lab= sitk.ReadImage(lab)

        '''
                    当数据是D*W*H的时候，这个数据的值可能是[1,NB_lable],所以需要进行np.where转换
                    当数据是D*W*H*C的时候，这个数据如果有通道C，则里面的数据保证为0,1
                    最后神经网络的输入数据为 D*W*H*1 ，在本网络中，支持的是这种数据
                    '''
        arr_mv_lab = sitk.GetArrayFromImage(sitk_mv_lab)
        arr_mv_lab = np.where(arr_mv_lab == self.args.component, 1, 0)
        arr_mv_lab=np.expand_dims(arr_mv_lab,axis=-1)
        return  arr_mv_img,arr_mv_lab

    def generate_sequnce_index(self):
        index_mv=self.index//len(self.img_fix)
        index_fix=self.index%len(self.img_fix)
        self.index=self.index+1
        self.index=self.index%(len(self.img_fix) * len(self.img_mv1))
        return  index_mv,index_fix
    def generate_random_index(self):
        return  np.random.randint(self.num),np.random.randint(self.num)

class Conven_Sampler(Sampler):

    def __init__(self):
        pass