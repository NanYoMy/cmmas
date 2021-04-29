import argparse
import numpy as np
parser = argparse.ArgumentParser(description='patch based embedding')
# parser.add_argument("--target_img", type=str, nargs=1, required=True, help="target image")
# parser.add_argument("--atlas_img_list", type=str, nargs='+', required=True, help="list of atlas images")
# parser.add_argument("--atlas_lab_list", type=str, nargs='+', required=True, help="list of atlas labelmaps")
# parser.add_argument("--out_file", type=str, nargs=1, required=True, help="output fusion results")
# parser.add_argument("--probabilities", action="store_true", help="compute segmentation probabilities")
# parser.add_argument("--patch_radius", type=str, nargs=1, help="image patch radius (default 3x3x3)")
# parser.add_argument("--search_radius", type=str, nargs=1, help="search neighborhood radius (default 1x1x1)")
# parser.add_argument("--fusion_radius", type=str, nargs=1, help="neighborhood fusion radius (default 1x1x1)")
# parser.add_argument("--struct_sim", type=float, nargs=1, default=[0.9],
#                     help="structural similarity threshold (default 0.9)")
# parser.add_argument("--normalization", type=str, nargs=1,
#                     help="patch normalization type [L2 | zscore | none] (default zscore)")
# parser.add_argument("--method", type=str, nargs=1, required=True, help="nlwv, nlbeta, deeplf, lasso")
# parser.add_argument("--regularization", type=float, nargs=1, default=[0.001],
#                     help="(nlwv, lasso, nlbeta) regularization parameter for label fusion method")
# parser.add_argument("--load_net", type=str, nargs=1, help="(deeplf) dirutil with the deep neural network")
# parser.add_argument("--label_grp", type=int, nargs='+', help="(optional) list of label ids to segment")
# parser.add_argument("--consensus_thr", type=float, nargs=1, default=[0.9],
#                     help="(optional) consensus threshold for creating segmentation mask (default 0.9)")
# parser.add_argument("--classification_metrics", type=str, nargs=1,
#                     help="compute classification metrics in non-consensus region (needs target labelmap)")
parser.add_argument("--model_dir", type=str, nargs=1, help="")
parser.add_argument("--dir_save", type=str, nargs=1, help="")
parser.add_argument("--test_atlas_target", type=str, nargs=1, help="")
parser.add_argument("--train_atlas_target", type=str, nargs=1, help="")
parser.add_argument("--patch_size", type=int, nargs=1, help="")
parser.add_argument("--thres", type=float, nargs=1, help="the threshold to decide weather two patches are same for preparing traning sample")
parser.add_argument("--consensus_thr", type=float, nargs=1, help="if consensus_thr*nb_atlas")
parser.add_argument("--n_support_sample", type=int, nargs=1, help="")
parser.add_argument("--n_query_sample", type=int, nargs=1, help="")

# args = parser.parse_args('--model_dir ../data/local/man_mr_ct_20_sim/model.ckpt '
#                          '--dir_save ../data/local/man_mr_ct_20_sim/ '
#                          '--test_atlas_target ../../data_vote_man/MR_CT/test/model_24000/*ct_train_%s_image_A_T/ '
#                          '--train_atlas_target ../../data_vote_man/MR_CT/train/model_24000/*ct_train_%s_image_A_T/ '
#                          '--patch_size 15 '
#                          '--n_support_sample 9 '
#                          '--n_query_sample 1 '
#                          '--consensus_thr 0.7 '
#                          '--thres 0.85 '.split())
args = parser.parse_args('--model_dir ../data/local/man_ct_mr_20_sim/model.ckpt '
                         '--dir_save ../data/local/man_ct_mr_20_sim/ '
                         '--test_atlas_target ../../data_vote_man/CT_MR/test/model_24000/*mr_train_%s_image_A_T/ '
                         '--train_atlas_target ../../data_vote_man/CT_MR/train/model_24000/*mr_train_%s_image_A_T/ '
                         '--patch_size 15 '
                         '--n_support_sample 6 '
                         '--n_query_sample 1 '
                         '--consensus_thr 0.7 '
                         '--thres 0.85 '.split())
import logging
level=logging.INFO
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
def get_args():
    return args
# args = parser.parse_args()
# EXAMPLES:
# args = parser.parse_args(
#     '--atlas_img_list ./testdata/atlas_img_mr_train_1001_image0.nii.gz ./testdata/atlas_img_mr_train_1002_image0.nii.gz   --target_img ./testdata/target_img_ct_train_1017_image0.nii.gz --classification_metrics ./testdata/target_lab_ct_train_1017_label0.nii.gz --atlas_lab_list ./testdata/atlas_lab_mr_train_1001_label0.nii.gz ./testdata/atlas_lab_mr_train_1002_label0.nii.gz --method nlwv --search_radius 1x1x1 --fusion_radius 1x1x1 --struct_sim 0.900000 --label_grp 1  --load_net ./grp0_epch0_007.dat --out_file ./testdata/out.nii.gz'.split())
