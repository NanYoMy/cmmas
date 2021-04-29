import numpy as np
import SimpleITK as sitk
import cv2
import os
from dirutil.helper import sort_glob
def drawContour(dcm_file, gt_label, pred_label,slice=48):

    image = sitk.ReadImage(dcm_file)
    image= sitk.RescaleIntensity(image)
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.squeeze(image_array)
    image_array = image_array.astype(np.float32)[:,slice,:]

    gt = sitk.ReadImage(gt_label)
    gt_array = sitk.GetArrayFromImage(gt)
    gt_array = np.squeeze(gt_array).astype(np.uint8)[:,slice,:]

    pred = sitk.ReadImage(pred_label)
    pred_array = sitk.GetArrayFromImage(pred)
    pred_array = np.squeeze(pred_array).astype(np.uint8)[:,slice,:]

    # 若不转化为彩色，那么最后画出来的contour也只能是灰度的
    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(
        gt_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    contours2, hierarchy2 = cv2.findContours(
        pred_array, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)

    cv2.drawContours(image_array, contours, -1,  (0,0, 255) , 1)
    cv2.drawContours(image_array, contours2, -1, (255, 0, 0), 1)
    image_array=np.flip(image_array,0)
    # image_array=cv2.resize(image_array,(256,256),cv2.INTER_LANCZOS4)
    cv2.imshow("liver_contour", image_array)
    cv2.waitKey()
    out_path=os.path.join(os.path.dirname(pred_label),"%s-%d.png"%(os.path.basename(pred_label),slice))
    cv2.imwrite(out_path,image_array)


'''


'''
if __name__=='__main__':
    pred_labels=sort_glob(r'E:\homework\cmmas-tmi\img\registration\ct-mr-1\*'+"\*label*.nii.gz")
    target_label=r"E:\homework\cmmas-tmi\img\registration\ct-mr-1\8_mr_target_label.nii.gz"
    target_img=r"E:\homework\cmmas-tmi\img\registration\ct-mr-1\8_mr_target_image.nii.gz"
    for lab in pred_labels:
        drawContour(target_img,target_label,lab)
