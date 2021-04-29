import SimpleITK as sitk
import numpy as np
import os
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from  sitkImageIO.itkdatawriter import sitk_write_image
OUTPUT_DIR='../tmp'
def parameter_space_regular_grid_sampling(*transformation_parameters):
    '''
    Create a list representing a regular sampling of the parameter space.
    Args:
        *transformation_paramters : two or more numpy ndarrays representing parameter values. The order
                                    of the arrays should match the ordering of the SimpleITK transformation
                                    parametrization (e.g. Similarity2DTransform: scaling, rotation, tx, ty)
    Return:
        List of lists representing the regular grid sampling.

    Examples:
        #parametrization for 2D translation transform (tx,ty): [[1.0,1.0], [1.5,1.0], [2.0,1.0]]
        >>>> parameter_space_regular_grid_sampling(np.linspace(1.0,2.0,3), np.linspace(1.0,1.0,1))
    '''
    return [[np.asscalar(p) for p in parameter_values]
            for parameter_values in np.nditer(np.meshgrid(*transformation_parameters))]

def parameter_space_random_sampling(thetaX,thetaY,tx,ty,scale,n=10):
    theta_x_vals = (thetaX[1] - thetaX[0]) * np.random.random(n) + thetaX[0]
    theta_y_vals = (thetaY[1] - thetaY[0]) * np.random.random(n) + thetaY[0]
    tx_vals = (tx[1] - tx[0]) * np.random.random(n) + tx[0]
    ty_vals = (ty[1] - ty[0]) * np.random.random(n) + ty[0]
    s_vals = (scale[1] - scale[0]) * np.random.random(n) + scale[0]
    res = list(zip(theta_x_vals, theta_y_vals,  tx_vals, ty_vals,  s_vals))
    '''
    这个函数不能使用，有bug
    '''
    return [list(eul2quat(*(p[0:2]))[0:2]) + list(p[2:5]) for p in res]

def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parametrization uses the vector portion of a versor we don't have an
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use, in radians.
        tx, ty, tz: numpy ndarrays with the translation values to use in mm.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0], parameter_values[1], parameter_values[2])) +
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in
            np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]


def similarity3D_parameter_space_random_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale, n):
    '''
    Create a list representing a random (uniform) sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parametrization uses the vector portion of a versor we don't have an
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: Ranges of Euler angle values to use, in radians.
        tx, ty, tz: Ranges of translation values to use in mm.
        scale: Range of scale values to use.
        n: Number of samples.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    theta_x_vals = (thetaX[1] - thetaX[0]) * np.random.random(n) + thetaX[0]
    theta_y_vals = (thetaY[1] - thetaY[0]) * np.random.random(n) + thetaY[0]
    theta_z_vals = (thetaZ[1] - thetaZ[0]) * np.random.random(n) + thetaZ[0]
    tx_vals = (tx[1] - tx[0]) * np.random.random(n) + tx[0]
    ty_vals = (ty[1] - ty[0]) * np.random.random(n) + ty[0]
    tz_vals = (tz[1] - tz[0]) * np.random.random(n) + tz[0]
    s_vals = (scale[1] - scale[0]) * np.random.random(n) + scale[0]
    res = list(zip(theta_x_vals, theta_y_vals, theta_z_vals, tx_vals, ty_vals, tz_vals, s_vals))
    return [list(eul2quat(*(p[0:3]))) + list(p[3:7]) for p in res]


def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r = np.zeros((3, 3))
    r[0, 0] = cz * cy
    r[0, 1] = cz * sy * sx - sz * cx
    r[0, 2] = cz * sy * cx + sz * sx

    r[1, 0] = sz * cy
    r[1, 1] = sz * sy * sx + cz * cx
    r[1, 2] = sz * sy * cx - cz * sx

    r[2, 0] = -sy
    r[2, 1] = cy * sx
    r[2, 2] = cy * cx

    # Compute quaternion:
    qs = 0.5 * np.sqrt(r[0, 0] + r[1, 1] + r[2, 2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs, 0.0, atol):
        i = np.argmax([r[0, 0], r[1, 1], r[2, 2]])
        j = (i + 1) % 3
        k = (j + 1) % 3
        w = np.sqrt(r[i, i] - r[j, j] - r[k, k] + 1)
        qv[i] = 0.5 * w
        qv[j] = (r[i, j] + r[j, i]) / (2 * w)
        qv[k] = (r[i, k] + r[k, i]) / (2 * w)
    else:
        denom = 4 * qs
        qv[0] = (r[2, 1] - r[1, 2]) / denom;
        qv[1] = (r[0, 2] - r[2, 0]) / denom;
        qv[2] = (r[1, 0] - r[0, 1]) / denom;
    return qv


def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters,
                           output_prefix, output_suffix,
                           interpolator=sitk.sitkLinear, default_intensity_value=0.0):
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''
    all_images = []  # Used only for display purposes in this notebook.
    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                  interpolator, default_intensity_value)
        # sitk.WriteImage(aug_image, output_prefix + '_' +
        #                 '_'.join(str(param) for param in current_parameters) + '_.' + output_suffix)

        all_images.append(aug_image)  # Used only for display purposes in this notebook.
    return all_images  # Used only for display purposes in this notebook.

def augment_img_lab(datas, labs, img_size=96):

    assert len(datas)==1
    assert len(labs)==1

    dimension = datas[0].GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)
    for img in datas:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    # Select arbitrary number of pixels per dimension, smallest size that yields desired results
    # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
    # often result in non-isotropic pixel spacing.
    reference_size = [img_size]*dimension
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    # Another possibility is that you want isotropic pixels, then you can specify the image size for one of
    # the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
    # spacing set accordingly.
    # Uncomment the following lines to use this strategy.
    #reference_size_x = 128
    #reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
    #reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    reference_image = sitk.Image(reference_size, datas[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_lab = sitk.Image(reference_size, labs[0].GetPixelIDValue())
    reference_lab.SetOrigin(reference_origin)
    reference_lab.SetSpacing(reference_spacing)
    reference_lab.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))


    ########################################################
    aug_transform = sitk.Similarity2DTransform() if dimension == 2 else sitk.Similarity3DTransform()

    all_images = []

    for img,lab in zip(datas,labs):
        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)

        # Set the augmenting transform's center so that rotation is around the image center.
        aug_transform.SetCenter(reference_center)

        if dimension == 2:
            # The parameters are scale (+-10%), rotation angle (+-10 degrees), x translation, y translation
            transformation_parameters_list = parameter_space_regular_grid_sampling(np.linspace(0.9, 1.1, 5),
                                                                                   np.linspace(-np.pi / 18.0, np.pi / 18.0,
                                                                                               5),
                                                                                   np.linspace(-10, 10, 5),
                                                                                   np.linspace(-10, 10, 5))
            tmp=np.random.randint(len(transformation_parameters_list))
            transformation_parameters_list=[transformation_parameters_list[tmp]]
            # transformation_parameters_list = parameter_space_random_sampling(
            #     thetaX=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaY=(-np.pi / 18.0, np.pi / 18.0),
            #     tx=(-10.0, 10.0),
            #     ty=(-10.0, 10.0),
            #     scale=(0.9, 1.1),
            #     n=1)
        else:
            # transformation_parameters_list = similarity3D_parameter_space_random_sampling(
            #     thetaX=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaY=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaZ=(-np.pi / 18.0, np.pi / 18.0),
            #     tx=(-10.0, 10.0),
            #     ty=(-10.0, 10.0),
            #     tz=(-10.0, 10.0),
            #     scale=(0.9, 1.1),
            #     n=1)
            transformation_parameters_list = similarity3D_parameter_space_random_sampling(
                thetaX=(-np.pi / 18.0, np.pi / 18.0),
                thetaY=(-np.pi / 18.0, np.pi / 18.0),
                thetaZ=(-np.pi / 18.0, np.pi / 18.0),
                tx=(-10.0, 10.0),
                ty=(-10.0, 10.0),
                tz=(-10.0, 10.0),
                scale=(0.9, 1.1),
                n=1)
            tmp = np.random.randint(len(transformation_parameters_list))
            transformation_parameters_list = [transformation_parameters_list[tmp]]
        #         transformation_parameters_list = similarity3D_parameter_space_regular_sampling(np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(0.9,1.1,3))

        generated_images = augment_images_spatial(img, reference_image, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'img_spatial_aug' ), 'nii.gz')
        generated_labs = augment_images_spatial(lab, reference_lab, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'lab_spatial_aug'), 'nii.gz',interpolator=sitk.sitkNearestNeighbor,default_intensity_value=0)

        # if dimension == 2:  # in 2D we join all of the images into a 3D volume which we use for display.
        #     all_images.append(sitk.JoinSeries(generated_images))
        return generated_images,generated_labs

def augment_multi_imgs_lab(c0s, des, t2s, labs, img_size=96):

    assert len(labs)==1

    dimension = c0s[0].GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)
    for c0 in c0s:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(c0.GetSize(), c0.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    # Select arbitrary number of pixels per dimension, smallest size that yields desired results
    # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
    # often result in non-isotropic pixel spacing.
    reference_size = [img_size]*dimension
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    # Another possibility is that you want isotropic pixels, then you can specify the image size for one of
    # the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
    # spacing set accordingly.
    # Uncomment the following lines to use this strategy.
    #reference_size_x = 128
    #reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
    #reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    reference_image = sitk.Image(reference_size, c0s[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_lab = sitk.Image(reference_size, labs[0].GetPixelIDValue())
    reference_lab.SetOrigin(reference_origin)
    reference_lab.SetSpacing(reference_spacing)
    reference_lab.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))


    ########################################################
    aug_transform = sitk.Similarity2DTransform() if dimension == 2 else sitk.Similarity3DTransform()

    all_images = []

    for c0,de,t2,lab in zip(c0s,des,t2s, labs):
        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(c0.GetDirection())
        transform.SetTranslation(np.array(c0.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(c0.TransformContinuousIndexToPhysicalPoint(np.array(c0.GetSize()) / 2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)

        # Set the augmenting transform's center so that rotation is around the image center.
        aug_transform.SetCenter(reference_center)

        if dimension == 2:
            # The parameters are scale (+-10%), rotation angle (+-10 degrees), x translation, y translation
            #注意这个代码会有导致程序很慢，因为一次性便利所有参数空间
            # transformation_parameters_list = parameter_space_regular_grid_sampling(np.linspace(0.8, 1.2, 10),
            #                                                                        np.linspace(-np.pi / 18.0, np.pi / 18.0,12),
            #                                                                        np.linspace(-15, 15, 10),
            #                                                                        np.linspace(-15, 15, 10))
            # tmp=np.random.randint(len(transformation_parameters_list))
            # transformation_parameters_list=[transformation_parameters_list[tmp]]

            range=np.linspace(0.8, 1.2, 20)
            scale=range[np.random.randint(len(range))]
            range = np.linspace(-np.pi / 18.0, np.pi / 18.0,24)
            rot = range[np.random.randint(len(range))]
            range = np.linspace(-15, 15, 20)
            deltax = range[np.random.randint(len(range))]
            range = np.linspace(-15, 15, 20)
            deltay = range[np.random.randint(len(range))]
            transformation_parameters_list=[[scale,rot,deltax,deltay]]
            # transformation_parameters_list = parameter_space_random_sampling(
            #     thetaX=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaY=(-np.pi / 18.0, np.pi / 18.0),
            #     tx=(-10.0, 10.0),
            #     ty=(-10.0, 10.0),
            #     scale=(0.9, 1.1),
            #     n=1)
        else:
            # transformation_parameters_list = similarity3D_parameter_space_random_sampling(
            #     thetaX=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaY=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaZ=(-np.pi / 18.0, np.pi / 18.0),
            #     tx=(-10.0, 10.0),
            #     ty=(-10.0, 10.0),
            #     tz=(-10.0, 10.0),
            #     scale=(0.9, 1.1),
            #     n=1)
            transformation_parameters_list = similarity3D_parameter_space_random_sampling(
                thetaX=(-np.pi / 18.0, np.pi / 18.0),
                thetaY=(-np.pi / 18.0, np.pi / 18.0),
                thetaZ=(-np.pi / 18.0, np.pi / 18.0),
                tx=(-10.0, 10.0),
                ty=(-10.0, 10.0),
                tz=(-10.0, 10.0),
                scale=(0.9, 1.1),
                n=1)
            tmp = np.random.randint(len(transformation_parameters_list))
            transformation_parameters_list = [transformation_parameters_list[tmp]]
        #         transformation_parameters_list = similarity3D_parameter_space_regular_sampling(np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(0.9,1.1,3))

        generated_images_c0 = augment_images_spatial(c0, reference_image, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'img_spatial_aug' ), 'nii.gz')

        generated_images_de = augment_images_spatial(de, reference_image, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'img_spatial_aug' ), 'nii.gz')

        generated_images_t2 = augment_images_spatial(t2, reference_image, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'img_spatial_aug' ), 'nii.gz')

        generated_labs = augment_images_spatial(lab, reference_lab, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'lab_spatial_aug'), 'nii.gz',interpolator=sitk.sitkNearestNeighbor,default_intensity_value=0)

        # if dimension == 2:  # in 2D we join all of the images into a 3D volume which we use for display.
        #     all_images.append(sitk.JoinSeries(generated_images))
        return generated_images_c0,generated_images_de,generated_images_t2,generated_labs

'''
对intensity和label图像同时使用相同的参数进行变化
'''
def augment_imgs_labs(imgs,labs, img_size=96):


    dimension = imgs[0].GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)
    for img in imgs:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    reference_size = [img_size]*dimension
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, imgs[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_lab = sitk.Image(reference_size, labs[0].GetPixelIDValue())
    reference_lab.SetOrigin(reference_origin)
    reference_lab.SetSpacing(reference_spacing)
    reference_lab.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    aug_transform = sitk.Similarity2DTransform() if dimension == 2 else sitk.Similarity3DTransform()
    ########################################################

    img=imgs[0]
    ####create spatial transform
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # Set the augmenting transform's center so that rotation is around the image center.
    aug_transform.SetCenter(reference_center)

    if dimension == 2:
        range = np.linspace(0.8, 1.2, 20)
        scale = range[np.random.randint(len(range))]
        range = np.linspace(-np.pi / 18.0, np.pi / 18.0, 24)
        rot = range[np.random.randint(len(range))]
        range = np.linspace(-15, 15, 20)
        deltax = range[np.random.randint(len(range))]
        range = np.linspace(-15, 15, 20)
        deltay = range[np.random.randint(len(range))]
        transformation_parameters_list = [[scale, rot, deltax, deltay]]
    else:
        transformation_parameters_list = similarity3D_parameter_space_random_sampling(
            thetaX=(-np.pi / 28.0, np.pi / 28.0),
            thetaY=(-np.pi / 28.0, np.pi / 28.0),
            thetaZ=(-np.pi / 28.0, np.pi / 28.0),
            tx=(-10.0, 10.0),
            ty=(-10.0, 10.0),
            tz=(-10.0, 10.0),
            scale=(0.85, 1.45),
            n=1)
        tmp = np.random.randint(len(transformation_parameters_list))
        transformation_parameters_list = [transformation_parameters_list[tmp]]
    ####


    trans_imgs=[]
    trans_labs=[]
    for img in imgs:
        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.

        gen_img= augment_images_spatial(img, reference_image, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'img_spatial_aug' ), 'nii.gz')
        trans_imgs.append(gen_img[0])

    for lab in labs:
        gen_lab= augment_images_spatial(lab, reference_lab, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'lab_spatial_aug'), 'nii.gz',interpolator=sitk.sitkNearestNeighbor,default_intensity_value=0)
        trans_labs.append(gen_lab[0])
    return  trans_imgs,trans_labs

def augment_img(datas):

    dimension = datas[0].GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)
    for img in datas:
        reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

    # Create the reference image with a zero origin, identity direction cosine matrix and dimension
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()

    # Select arbitrary number of pixels per dimension, smallest size that yields desired results
    # or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
    # often result in non-isotropic pixel spacing.
    reference_size = [256]*dimension
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    # Another possibility is that you want isotropic pixels, then you can specify the image size for one of
    # the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
    # spacing set accordingly.
    # Uncomment the following lines to use this strategy.
    #reference_size_x = 128
    #reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
    #reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

    reference_image = sitk.Image(reference_size, datas[0].GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)


    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the
    # spacing will not yield the correct coordinates resulting in a long debugging session.
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))


    ########################################################
    aug_transform = sitk.Similarity2DTransform() if dimension == 2 else sitk.Similarity3DTransform()

    all_images = []

    for img in datas:
        # Transform which maps from the reference_image to the current img with the translation mapping the image
        # origins to each other.
        transform = sitk.AffineTransform(dimension)
        transform.SetMatrix(img.GetDirection())
        transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
        # Modify the transformation to align the centers of the original and reference image instead of their origins.
        centering_transform = sitk.TranslationTransform(dimension)
        img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize()) / 2.0))
        centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
        centered_transform = sitk.Transform(transform)
        centered_transform.AddTransform(centering_transform)

        # Set the augmenting transform's center so that rotation is around the image center.
        aug_transform.SetCenter(reference_center)

        if dimension == 2:
            # The parameters are scale (+-10%), rotation angle (+-10 degrees), x translation, y translation
            transformation_parameters_list = parameter_space_regular_grid_sampling(np.linspace(0.9, 1.1, 5),
                                                                                   np.linspace(-np.pi / 18.0, np.pi / 18.0,5),
                                                                                   np.linspace(-10, 10, 5),
                                                                                   np.linspace(-10, 10, 5))
            tmp=np.random.randint(len(transformation_parameters_list))
            transformation_parameters_list=[transformation_parameters_list[tmp]]
            # transformation_parameters_list = parameter_space_random_sampling(
            #     thetaX=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaY=(-np.pi / 18.0, np.pi / 18.0),
            #     tx=(-10.0, 10.0),
            #     ty=(-10.0, 10.0),
            #     scale=(0.9, 1.1),
            #     n=1)
        else:
            # transformation_parameters_list = similarity3D_parameter_space_random_sampling(
            #     thetaX=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaY=(-np.pi / 18.0, np.pi / 18.0),
            #     thetaZ=(-np.pi / 18.0, np.pi / 18.0),
            #     tx=(-10.0, 10.0),
            #     ty=(-10.0, 10.0),
            #     tz=(-10.0, 10.0),
            #     scale=(0.9, 1.1),
            #     n=1)
            transformation_parameters_list = similarity3D_parameter_space_random_sampling(
                thetaX=(-np.pi / 18.0, np.pi / 18.0),
                thetaY=(-np.pi / 18.0, np.pi / 18.0),
                thetaZ=(-np.pi / 18.0, np.pi / 18.0),
                tx=(-10.0, 10.0),
                ty=(-10.0, 10.0),
                tz=(-10.0, 10.0),
                scale=(0.9, 1.1),
                n=1)
        #         transformation_parameters_list = similarity3D_parameter_space_regular_sampling(np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-np.pi/18.0,np.pi/18.0,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(-10,10,3),
        #                                                                                        np.linspace(0.9,1.1,3))

        generated_images = augment_images_spatial(img, reference_image, centered_transform,
                                                  aug_transform, transformation_parameters_list,
                                                  os.path.join(OUTPUT_DIR, 'img_spatial_aug' ), 'nii.gz')

        # if dimension == 2:  # in 2D we join all of the images into a 3D volume which we use for display.
        #     all_images.append(sitk.JoinSeries(generated_images))
        return generated_images

def threshold_based_crop(image,inside_value=0,outside_value=255):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
    """
    # Set pixels that are in [min_intensity,otsu_threshold] to inside_value, values above otsu_threshold are
    # set to outside_value. The anatomy has higher intensity values than the background, so it is outside.
    # inside_value = 0
    # outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    # The bounding box's first "dim" entries are the starting index and last "dim" entries the size
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box) / 2):],
                                 bounding_box[0:int(len(bounding_box) / 2)])


if __name__=="__main__":
    import glob
    img_paths=glob.glob("../../datasets/myo_data/train25_myops_crop/*.nii.gz")
    lab_paths=glob.glob("../../datasets/myo_data/train25_myops_gd_crop/*.nii.gz")
    for p1,p2 in zip(img_paths,lab_paths):
        img = sitk.ReadImage(p1)
        lab = sitk.ReadImage(p2)
        # sitk_write_image(img,dir=OUTPUT_DIR,name='thres.nii.gz')
        augment_img_lab([img], [lab],96)














