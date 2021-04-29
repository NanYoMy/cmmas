import os
import shutil
from dirutil.helper import mkdir_if_not_exist,mkcleardir
import glob
def split(all_dir,test_dir,num):

    files = os.listdir(all_dir)
    files.sort()

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        os.makedirs(test_dir)
    else:
        os.makedirs(test_dir)

    for i in range(num):
        tmp=len(files)-1-i
        print(files[tmp])
        shutil.move(os.path.join(all_dir,files[tmp]),os.path.join(test_dir,files[tmp]))

def isVoteTestDir(name, atlas, target):
    for i in atlas:
        for j in target:
            if name.find(str(i))!=-1 and name.find(str(j))!=-1:
                return True
    return False



def split_vote(all_dir,test_dir,atlas,target):
    mkcleardir(test_dir)
    dir = os.listdir(all_dir)

    for model_dir in dir:
        samples=os.listdir(os.path.join(all_dir,model_dir))
        mkdir_if_not_exist(os.path.join(test_dir, os.path.basename(model_dir)))
        for item in samples:
            if isVoteTestDir(item, atlas, target):
                shutil.move(os.path.join(all_dir,model_dir,item), os.path.join(test_dir,model_dir))

def remove_file(test_dir,tag='_T_A'):
    dir = glob.glob(test_dir+"/*")
    for model_dir in dir:
        sub_dir=glob.glob(model_dir+"/*")
        for item in sub_dir:
            base_name=os.path.basename(item)
            if base_name.find('T_A')!=-1:
                shutil.rmtree(item)

if __name__=="__main__":
    split("E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-image_result_pre_reg",
          "E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-image_result_pre_reg_test",5)
    split("E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-image_result_pre_reg",
          "E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-image_result_pre_reg_test", 5)
    split("E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label_result_pre_reg",
          "E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label_result_pre_reg_test", 5)
    split("E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-label_result_pre_reg",
          "E:\MIA_CODE_DATA\zhuang_data\MMWHS\CT\\train\ct-label_result_pre_reg_test", 5)