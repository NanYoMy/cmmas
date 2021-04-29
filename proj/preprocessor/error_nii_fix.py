
import numpy as np
import nibabel as nib

def fix_image():
    fileobj=nib.load("E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-image\mr_train_1002_image.nii.gz")
    tmp=fileobj.dataobj
    new_tmp=tmp[10:,10:,10:]
    img = nib.Nifti1Image(new_tmp, fileobj.affine)
    nib.save(img,"E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-image\mr_train_1002_image_1.nii.gz")

    fileobj=nib.load("E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label\mr_train_1002_label.nii.gz")
    tmp=fileobj.dataobj
    new_tmp=tmp[10:,10:,10:]
    img = nib.Nifti1Image(new_tmp, fileobj.affine)
    nib.save(img,"E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label\mr_train_1002_label_1.nii.gz")

def fix_label():
    fileobj=nib.load("E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label\mr_train_1010_label.nii.gz")
    data=fileobj.dataobj[0:,0:,0:]
    tmp=np.copy(data)
    print(np.sum(tmp==421))
    print(np.unique(tmp))

    tmp[np.where(tmp==421)]=420
    img = nib.Nifti1Image(tmp, fileobj.affine)
    nib.save(img, "E:\MIA_CODE_DATA\zhuang_data\MMWHS\MRI\mr-label\mr_train_1010_label_1.nii.gz")


if __name__=="__main__":
    # fix_image()
    # fix_label()
    pass