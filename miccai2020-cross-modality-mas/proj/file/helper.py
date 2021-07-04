import shutil
import os
import time
def mkdir(out_put_dir):
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

def clear(rootdir):
    filelist = os.listdir(rootdir)  # 列出该目录下的所有文件名
    for f in filelist:
        filepath = os.path.join(rootdir, f)  # 将文件名映射成绝对路劲
        if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
            os.remove(filepath)  # 若为文件，则直接删除
            print(str(filepath) + " removed!")
        elif os.path.isdir(filepath):
            shutil.rmtree(filepath, True)  # 若为文件夹，则删除该文件夹及文件夹内所有文件
            print("dir " + str(filepath) + " removed!")

    # os.rmdir(rootdir)

def mk_or_cleardir(out_put_dir):
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    else:
        clear(out_put_dir)
def mkcleardir(out_put_dir):
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)
    else:
        clear(out_put_dir)
        # os.makedirs(out_put_dir)
        # shutil.rmtree(out_put_dir)
        shutil.rmtree(out_put_dir, True)  # 最后删除img总文件夹
        time.sleep(3)
        os.makedirs(out_put_dir)

def mkoutputname(dir,modelId,atlasId,targetId,type):
    return os.path.join(dir,"%s_atlas_%s_target_%s_%s"%(modelId,atlasId,targetId,type))


def filename(path,extension=False):

    tmp=os.path.basename(path)
    terms=tmp.split('.')
    return terms[0]
def listdir(dir):
    tmp=os.listdir(dir)
    tmp.sort()
    # tmp=tmp.sort()
    target_files = [os.path.join(dir, i) for i in tmp]
    return  target_files

def writeListToFile(file_list,out_put_file):
    if os.path.exists(out_put_file):
        os.remove(out_put_file)
    f=open(out_put_file,'w')

    f.writelines([line+'\n' for line in file_list])
    f.close()
