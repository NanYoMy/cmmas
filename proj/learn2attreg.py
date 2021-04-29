import argparse

from evaluate.metric import cross_validate

import os
import tensorflow as tf
tf.reset_default_graph()
tf.set_random_seed(17)
from numpy.random import seed
seed(17)
import random
random.seed(17)
os.environ['PYTHONHASHSEED']=str(17)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
from learn2reg.regnet import LabAttentionReg
from dirutil.helper import mkdir_if_not_exist,mk_or_cleardir
from learn2reg.prepare_mmwhs import prepare_mmwhs_reg_working_data,prepare_chaos_reg_working_data

from config.Defines import Get_Name_By_Index
from learn2reg.postprepare_mmwhs import post_process

MOLD_ID='mmwhs'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--iteration', dest='iteration', type=int, default=1503, help='# train iteration')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
parser.add_argument('--image_size', dest='image_size', type=int, default=96, help='the size of image_size')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--decay_fyeq', dest='decay_freq', type=int, default=100, help='decay the learning rate accoding to the iteration step')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=100, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using reidule block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=bool, default=True, help='gan loss defined in lsgan')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=4, help='miccai:32,')
parser.add_argument('--lambda_ben', dest='lambda_ben', type=float, default=0.2, help='momentum term of adam')


parser.add_argument('--Tatlas', dest='Tatlas',  default='mr', help='')
parser.add_argument('--Ttarget', dest='Ttarget', default='ct', help='')
# parser.add_argument('--component', dest='component', type=int, default=205, help='mmwhs 205=myocardium 500=lv; chaos 1=liver')
# parser.add_argument('--components', dest='components',  default='205,500', help='the label used for training model, the lv and myo could be trained jointly')
parser.add_argument('--component', dest='component', type=int, default=1, help='mmwhs 205=myocardium 500=lv; chaos 1=liver')
parser.add_argument('--components', dest='components',  default='1', help='the label used for training model, the lv and myo could be trained jointly')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--lambda_consis', dest='lambda_consis', type=float, default=0.1, help='momentum term of adam')
parser.add_argument('--phase', dest='phase', default='train', help='validate,train,test,trainSim,testSim,gen,post')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# of images in batch')
parser.add_argument('--task', dest='task', default='CHAOS', help='MMWHS,CHAOS')
parser.add_argument('--fold', dest='fold', type=int, default=1, help='fold')

parser.add_argument("--gen_num",dest='gen_num', type=int, nargs=1,default=3000, help="")
#query patch
parser.add_argument("--patch_size", dest='patch_size',type=int, nargs=2,default=15, help="")
parser.add_argument("--nb_patch", dest='nb_patch',type=int, nargs=1,default=5, help="")
parser.add_argument("--search_range", dest='search_range',type=int, nargs=1,default=10, help="")
parser.add_argument("--thres",dest='thres', type=float, nargs=1, default=0.85, help="the threshold to decide weather two patches are same for preparing traning sample")
parser.add_argument("--consensus_thr", dest='consensus_thr',type=float, nargs=1,default=0.7, help="if consensus_thr*nb_atlas")#the threshold for consensus
parser.add_argument("--n_support_sample",dest='n_support_sample', type=int, nargs=1,default=6, help="")
parser.add_argument("--n_query_sample",dest='n_query_sample', type=int, nargs=1,default=1, help="")
parser.add_argument("--sim_iteration",dest='sim_iteration', type=int, nargs=1,default=30000, help="")
args = parser.parse_args()

def globel_setup():
    global  args
    print("global set %s %s #### component %d ##### %s" % (args.Tatlas, args.Ttarget, args.component,Get_Name_By_Index(args.component)))
    MOLD_ID = "attentionreg_%s-%s-%s-%d-fold-%d-consis-%f" % (args.task, args.Tatlas, args.Ttarget, args.component,args.fold,args.lambda_consis)
    MOLD_ID_TEMPLATE = "attentionreg_%s-%s-%s-%d-fold-%s-consis-%f" % (args.task, args.Tatlas, args.Ttarget, args.component,"#",args.lambda_consis)
    DATA_ID = "%s-%s-%s-%d" % (args.task, args.Tatlas, args.Ttarget, args.component)

    parser.add_argument('--MOLD_ID_TEMPLATE', dest='MOLD_ID_TEMPLATE', default=MOLD_ID_TEMPLATE,help='')
    parser.add_argument('--MOLD_ID', dest='MOLD_ID', default=MOLD_ID,help='')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='../outputs/%s/checkpoint' % (MOLD_ID),help='models are saved here')
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='../datasets/%s' % (DATA_ID),
                        help='path of the dataset')
    parser.add_argument('--sample_dir', dest='sample_dir', default='../outputs/%s/sample' % (MOLD_ID),
                        help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='../outputs/%s/test' % (MOLD_ID),
                        help='test sample are saved here')
    parser.add_argument('--log_dir', dest='log_dir', default='../outputs/%s/log' % (MOLD_ID), help='log dir')
    parser.add_argument('--gated_att_dir', dest='gated_att_dir', default='../outputs/%s/gated_att/' % (MOLD_ID), help='log dir')
    parser.add_argument('--gen_dir', dest='gen_dir', default='../datasets/fusion_%s/' % (MOLD_ID), help='gen dir')
    parser.add_argument('--res_excel', dest='res_excel', default='../outputs/result/%s.xls'%(MOLD_ID_TEMPLATE),help='train,test,trainSim,testSim,gen,post')

    # parser.add_argument('--fusion_train', dest='fusion_train', default='../datasets/%s/fusion_train' % (MOLD_ID),help='path of the similariyt dataset')
    # parser.add_argument('--fusion_validate', dest='fusion_validate', default='../datasets/%s/fusion_validate' % (MOLD_ID),help='path of the similariyt dataset')

    args = parser.parse_args()

'''
biregnet有attention
'''
def main(_):

    globel_setup()
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        if args.phase == 'train':
            mkdir_if_not_exist(args.log_dir)
            # mk_or_cleardir(args.checkpoint_dir)
            mk_or_cleardir(args.log_dir)
            if args.task=='CHAOS':
                prepare_chaos_reg_working_data(args)
            else:
                prepare_mmwhs_reg_working_data(args)
            model = LabAttentionReg(sess, args)
            model.train()
        elif args.phase=='validate':
            # mk_or_cleardir(args.sample_dir)
            model = LabAttentionReg(sess, args)
            # model.show_gated_info()
            model.validate()
            # cross_validate(args)
        # elif args.phase=='test':
        #     mk_or_cleardir(args.test_dir)
        #     # prepare_data()
        #     model =LabAttentionReg (sess, args)
        #     model.test()
        #     post_process(args)
        # elif args.phase=='post':
        #     post_process(args)
        elif args.phase=='gen':
            #在../datasets/sim_ct_mr_**中生成数据
            # mk_or_cleardir(args.fusion_dataset_dir)
            # mk_or_cleardir(args.gen_dir)
            mk_or_cleardir(args.sample_dir)
            model = LabAttentionReg(sess, args)
            # model.generate()
            # model.validate()
            #同时生成验证和训练的数据
            model.generate_4_fusion()
        # elif args.phase=='trainSim':
        #     mk_or_cleardir(args.sim_checkpoint_dir)
        #     mk_or_cleardir(args.sim_sample_dir)
        #     mk_or_cleardir(args.sim_log_dir)
        #     model=PatchEmbbeding(sess,args)
        #     model.train()
        # elif args.phase=='testSim':
        #     mk_or_cleardir(args.sim_sample_dir)
        #     mk_or_cleardir(args.sim_test_dir)
        #     model = PatchEmbbeding(sess, args)
        #     model.test()
        # elif args.phase == 'fusion':
        #     # 调用生成label
        #     # 进行融合
        #     ngf = NGFFusion(args)
        #     ngf.run()
        #     # mvfusion=MVFusion(args)
        #     # mvfusion.run()
        #     mvfusion = SitkSTAPLEFusion(args)
        #     mvfusion.run()
        #     mvfusion = SitkMVFusion(args)
        #     mvfusion.run()
        elif args.phase=='summary':
            cross_validate(args)
        else:
            print("undefined phase")

if __name__ == '__main__':
    # test_code(args)
    tf.app.run()
