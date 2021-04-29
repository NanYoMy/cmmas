
Debug=False
if Debug:
    evaluete_sample=1
else:
    evaluete_sample=20
    # 需要用于测试的样本
fusion_sample=1
atlas_sample=evaluete_sample

# Label_Index = [500, 600, 420, 550, 205, 820, 850]
LABEL_INDEX=5
LABEL_VALUE=205
# LABEL_INDEX=1
# LABEL_VALUE=500
'''
'''
#votenet's atlas and target
Label_Index = [500, 205,600, 420, 550, 820, 850]
'''
91.7 79.1 86.7 86.9 86.6  85.6 
'''
Label_name = ["left ventricle","myocardium", " right ventricle", "left atrium", "right atrium",
              "ascending aorta", "pulmonary artery"]
def Get_Name_By_Index(index):
    for i,L in enumerate(Label_Index):
        if L==index:
            return Label_name[i]
    return "undefined_component_index"



