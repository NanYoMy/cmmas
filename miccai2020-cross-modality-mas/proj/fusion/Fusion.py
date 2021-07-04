'''
joint label fusion framework
'''
import config.configer
from dataprocessor.itkdatareader import FusionSitkDataReader
from fusion.entropyhelper import conditional_entropy_target_label
import os
import SimpleITK as sitk
from config.Defines import atlas_sample
from fusion.fusionhelper import combineLabels, computeWeightMap,processProbabilityImage
from file.helper import mkdir,listdir,writeListToFile,filename
reg_config = config.configer.get_reg_config()
class JLF:
    def __init__(self, atlas_lab_dir_base=None,atlas_img_dir_base=None,target_dir=None):
        if target_dir:
            target_files = [os.path.join(target_dir,i) for i in os.listdir(target_dir)]
            self.targets=FusionSitkDataReader(target_files)
        self.lab_dir_base=os.path.dirname(atlas_lab_dir_base)

        self.atlas_dir=[]
        self.lab_files=[]
        self.img_files=[]
        self.numb_atlas=atlas_sample

        for i in range(0,atlas_sample):
            tmp_atlas_dir=atlas_lab_dir_base+"//"+str(i)
            self.lab_files.append([os.path.join(tmp_atlas_dir,i)  for i in os.listdir(tmp_atlas_dir) if ( ('.nii.gz' in i) and ('label' in i) )])
            tmp_atlas_dir=atlas_img_dir_base+"//"+str(i)
            # self.atlas_dir=self.atlas_dir+[tmp_atlas_dir]
            self.img_files.append([os.path.join(tmp_atlas_dir,i)  for i in os.listdir(tmp_atlas_dir) if ( ('.nii.gz' in i) and ('image' in i) )])

    def fusion(self):
        pass

    # def mkdir(self,out_put_dir):
    #     if not os.path.exists(out_put_dir):
    #         os.makedirs(out_put_dir)

    def run(self):
        for i in range(self.targets.num_data):
            target=self.targets.get_data(i)
            # rank
            res=self.rank_atlas( i, target)
            atlas_img_candi=[self.img_files[j][i] for j in  res]
            atlas_lab_candi=[self.lab_files[j][i] for j in  res]
            fusion_atlas_img=FusionSitkDataReader(atlas_img_candi)
            fusion_atlas_lab=FusionSitkDataReader(atlas_lab_candi)
            #fusion

            weightMapDict = {}
            labelListDict = {}
            targetImage=self.targets.get_file_obj(i)
            for movingId in range(0,fusion_atlas_img.num_data):
                movingImage = fusion_atlas_img.get_file_obj(movingId)
                # moving image
                weightMap = computeWeightMap(targetImage, movingImage, voteType="Local")
                weightMapDict[movingId] = weightMap
                labelListDict[movingId] = {}
                # for structureName in range(0,fusion_atlas_lab.num_data):
                labelListDict[movingId][reg_config['Data']['structure']] = fusion_atlas_lab.get_file_obj(movingId)

            combinedLabelDict = combineLabels(weightMapDict, labelListDict)

            for structureName in [reg_config['Data']['structure']]:
                probabilityImage = combinedLabelDict[structureName]
                binaryImage = processProbabilityImage(probabilityImage)
                mkdir(self.lab_dir_base+"//fusion")
                # sitk.WriteImage(probabilityImage,self.lab_dir_base + "//fusion/"+self.targets.get_file_name(i).replace("image", "label_prob"))
                sitk.WriteImage(binaryImage,self.lab_dir_base + "//fusion/"+self.targets.get_file_name(i).replace("image", "label"))
                # sitk_write_image(binaryImage, self.targets.get_file_obj(i),
                #                  self.lab_dir_base + "/20/", )

                # sitk_write_image(binaryImage, self.targets.get_file_obj(i),
                #                  self.lab_dir_base + "/20/", self.targets.get_file_name(i).replace("image", "label"))
            # 写出fusion的结果
            # fusion_atlas=np.where(sum>=self.numb_atlas/2,1,0)
            # sitk_write_image(fusion_atlas,self.targets.get_file_obj(i),
            #                  self.lab_dir_base+"/20/",self.targets.get_file_name(i).replace("image","label"))

    def argsort(self,seq):
        return [x for x, y in sorted(enumerate(seq), key=lambda x: x[1])]
    def fuse_lab(self,i,candidate_index):
        for j in candidate_index :
            tmp_labels = FusionSitkDataReader(self.lab_dir_base + "/" + str(j) + "")
            #获取有用的atlas
            atlas_lab = tmp_labels.get_data(i)


    def rank_atlas(self, i, target):
        entropy_cond=[]
        for j in range(0, self.numb_atlas):
            tmp_labels = FusionSitkDataReader(self.lab_files[j])
            atlas_lab = tmp_labels.get_data(i)
            entropy_cond = entropy_cond + [conditional_entropy_target_label(target, atlas_lab, [100, 2])]
        rank=self.argsort(entropy_cond)
        return rank[0:5]

class zxhJLF():
    def __init__(self,reg_base_dir,target_dir):
        self.label=reg_base_dir+"//lab//"
        self.img=reg_base_dir+"//img//"
        self.target_dir=target_dir
        self.fusion_dir=reg_base_dir+"//lab//fusion//"
    def run(self):
        target_files=listdir(self.target_dir)

        for target_i in range(0,40):
            lab_files=[]
            img_files=[]
            for atlas_j in range(0,20):
                label_dir=listdir(self.label+str(atlas_j)+"//")
                img_dir=listdir(self.img+str(atlas_j)+"//")
                lab_files=lab_files+[label_dir[target_i]]
                img_files=img_files+[img_dir[target_i]]
            target_file = target_files[target_i]
            lab_txt="lab.txt"
            img_txt="img.txt"
            mkdir(self.fusion_dir)
            fusion_out=self.fusion_dir+filename(target_file).replace("image","label")
            writeListToFile(lab_files,lab_txt)
            writeListToFile(img_files,img_txt)
            cmd=".\zxhtool\zxhLabelFuse.exe %s %s %s %s %s -method1"%(target_file,fusion_out,str(20),img_txt,lab_txt)
            rv=os.system(cmd)
            print(rv)



class SVW_JLF(JLF):
    def __init__(self):
        pass

