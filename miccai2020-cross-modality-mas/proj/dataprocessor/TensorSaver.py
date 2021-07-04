
import nibabel as nib

def saveTensor(tensor,path):
    batch_num=tensor.shape[0]
    for i in range(batch_num):
        img=nib.Nifti1Image(tensor[i,:,:,:,0],affine=None)
        nib.save(img,"%s_%d.nii.gz"%(path,i))