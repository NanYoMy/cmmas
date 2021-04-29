import argparse
import tensorflow as tf
tf.set_random_seed(19)
from numpy.random import seed
seed(17)

from dirutil.helper import mkdir_if_not_exist,mk_or_cleardir
from learn2reg.prepare_mmwhs import prepare_mmwhs_reg_working_data

from config.Defines import Get_Name_By_Index
from learn2reg.postprepare_mmwhs import post_process


from fusion.fusionnetworks import LabIntSimNet

MOLD_ID='mmwhs'
parser = argparse.ArgumentParser(description='')
from evaluate.metric import cross_validate
parser.add_argument('--iteration', dest='iteration', type=int, default=50001, help='# train iteration')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--n_label', dest='n_label', type=int, default=3, help='# of label class')
parser.add_argument('--image_size', dest='image_size', type=int, default=96, help='the size of image_size')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=3, help='the size of image_size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help='initial learning rate for adam')
parser.add_argument('--decay_fyeq', dest='decay_freq', type=int, default=1000, help='decay the learning rate accoding to the iteration step')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=100, help='save a model every save_freq iterations')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=100, help='print the debug information every print_freq iterations')
parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
parser.add_argument('--num_channel_initial', dest='num_channel_initial', type=int, default=16, help='miccai:32,')
parser.add_argument('--lambda_consis', dest='lambda_consis', type=float, default=0.1, help='momentum term of adam')

parser.add_argument('--Tatlas', dest='Tatlas',  default='ct', help='')
parser.add_argument('--Ttarget', dest='Ttarget', default='mr', help='')
parser.add_argument('--task', dest='task', default='MMWHS', help='MMWHS,CHAOS')
parser.add_argument('--fold', dest='fold', type=int, default=1, help='the size of image_size')
parser.add_argument('--res_excel', dest='res_excel', default='../outputs/result/fusion_result.xls', help='')
parser.add_argument('--component', dest='component', type=int, default=205, help='205=myocardium 500=lv')
parser.add_argument('--components', dest='components',  default='205', help='both take account two label during the training of reg-net')
parser.add_argument('--phase', dest='phase', default='fusion', help='train,validate,')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')

args = parser.parse_args()


def globel_setup():
    global  args
    print("global set %s %s #### component %d ##### %s" % (args.Tatlas, args.Ttarget, args.component,Get_Name_By_Index(args.component)))
    #数据源于attreg的结果
    DATA_ID = "attentionreg_%s-%s-%s-%d-fold-%d-consis-%f" % (args.task, args.Tatlas, args.Ttarget, args.component,args.fold,args.lambda_consis)

    # dataset_dir/fusion_train_atlas_label就是训练数据, 而 dataset_dir/atlas_label是验证数据
    parser.add_argument('--dataset_dir', dest='dataset_dir', default='../outputs/%s/sample' % (DATA_ID),help='path of the dataset')

    MOLD_ID = "labelIntensityfusionnet_%s-%s-%s-%d-fold-%d" % (args.task,args.Tatlas, args.Ttarget, args.component,args.fold)
    MOLD_ID_TEMPLATE = "labelIntensityfusionnet_%s-%s-%s-%d-fold-%s" % (args.task,args.Tatlas, args.Ttarget, args.component,'#')
    parser.add_argument('--MOLD_ID_TEMPLATE', dest='MOLD_ID_TEMPLATE', default=MOLD_ID_TEMPLATE,help='')
    parser.add_argument('--MOLD_ID', dest='MOLD_ID', default=MOLD_ID,help='models are saved here')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='../outputs/%s/checkpoint' % (MOLD_ID),help='models are saved here')
    parser.add_argument('--sample_dir', dest='sample_dir', default='../outputs/%s/sample' % (MOLD_ID),help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='../outputs/%s/test' % (MOLD_ID),help='test sample are saved here')
    parser.add_argument('--log_dir', dest='log_dir', default='../outputs/%s/log' % (MOLD_ID), help='log dir')
    parser.add_argument('--gen_dir', dest='gen_dir', default='../datasets/labeldenoise_%s/' % (MOLD_ID), help='log dir')
    parser.add_argument('--validate_dir', dest='validate_dir', default='../outputs/%s/validate/' % (MOLD_ID), help='log dir')

    args = parser.parse_args()


def main(_):

    globel_setup()
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        if args.phase == 'train':
            print(args.log_dir)
            mkdir_if_not_exist(args.log_dir)
            mk_or_cleardir(args.log_dir)
            model = LabIntSimNet(sess, args)
            model.train()
        elif args.phase=='validate':
            mk_or_cleardir(args.sample_dir)
            model = LabIntSimNet(sess, args)
            model.validate()
        elif args.phase=='summary':
            cross_validate(args)
            # model=LabIntSimNet(sess,args)
            # model.cross_validate()
        elif args.phase=='test':
            mk_or_cleardir(args.test_dir)
            model = LabIntSimNet(sess, args)
            model.test()
        elif args.phase=='fusion':
            mk_or_cleardir(args.validate_dir)
            model = LabIntSimNet(sess, args)
            model.fusion()
        else:
            print("undefined phase")

if __name__ == '__main__':
    # test_code(args)
    tf.app.run()