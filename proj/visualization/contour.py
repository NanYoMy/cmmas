'''
https://blog.csdn.net/HOMEGREAT/article/details/102689718
'''
import cv2
import numpy as np
def union_image_mask(image_path, mask_path, num):
    # 读取原图
    image = cv2.imread(image_path)
    # print(image.shape) # (400, 500, 3)
    # print(image.size) # 600000
    # print(image.dtype) # uint8

    # 读取分割mask，这里本数据集中是白色背景黑色mask
    mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 裁剪到和原图一样大小
    mask_2d = mask_2d[0:400, 0:500]
    h, w = mask_2d.shape
    cv2.imshow("2d", mask_2d)

    # 在OpenCV中，查找轮廓是从黑色背景中查找白色对象，所以要转成黑色背景白色mask
    mask_3d = np.ones((h, w), dtype='uint8')*255
    # mask_3d_color = np.zeros((h,w,3),dtype='uint8')
    mask_3d[mask_2d[:, :] == 255] = 0
    cv2.imshow("3d", mask_3d)
    ret, thresh = cv2.threshold(mask_3d, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    cv2.drawContours(image, [cnt], 0, (0, 255, 0), 1)
    # 打开画了轮廓之后的图像
    cv2.imshow('mask', image)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    # 保存图像
    # cv2.imwrite("./image/result/" + str(num) + ".bmp", image)

from PIL import Image, ImageFilter
import numpy as np

def drawContour(m,s,c,RGB):
    """Draw edges of contour 'c' from segmented image 's' onto 'm' in colour 'RGB'"""
    # Fill contour "c" with white, make all else black
    thisContour = s.point(lambda p:p==c and 255)
    # DEBUG: thisContour.save(f"interim{c}.png")

    # Find edges of this contour and make into Numpy array
    thisEdges   = thisContour.filter(ImageFilter.FIND_EDGES)
    thisEdgesN  = np.array(thisEdges)

    # Paint locations of found edges in color "RGB" onto "main"
    m[np.nonzero(thisEdgesN)] = RGB
    return m

if __name__ == "__main__":
    # Load segmented image as greyscale
    seg = Image.open('../../datasets/visual/segmented.png').convert('L')
    # Load main image - desaturate and revert to RGB so we can draw on it in colour
    main = Image.open('../../datasets/visual/main.png').convert('L').convert('RGB')
    mainN = np.array(main)

    mainN = drawContour(mainN, seg, 1, (255, 0, 0))  # draw contour 1 in red
    mainN = drawContour(mainN, seg, 2, (255, 255, 0))  # draw contour 2 in yellow
    # Save result
    Image.fromarray(mainN).save('result.png')
