import SimpleITK as sitk
import numpy as np


#intensity_augmented_images = augment_images_intensity(data, os.path.join(OUTPUT_DIR, 'intensity_aug'), 'mha')
def augment_images_intensity(image_list, output_prefix, output_suffix):
    '''
    Generate intensity modified images from the originals.
    Args:
        image_list (iterable containing SimpleITK images): The images which we whose intensities we modify.
        output_prefix (string): output file name prefix (file name: output_prefixi_FilterName.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefixi_FilterName.output_suffix).
    '''

    # Create a list of intensity modifying filters, which we apply to the given images
    filter_list = []

    # Smoothing filters

    filter_list.append(sitk.SmoothingRecursiveGaussianImageFilter())
    filter_list[-1].SetSigma(2.0)

    filter_list.append(sitk.DiscreteGaussianImageFilter())
    filter_list[-1].SetVariance(4.0)

    filter_list.append(sitk.BilateralImageFilter())
    filter_list[-1].SetDomainSigma(4.0)
    filter_list[-1].SetRangeSigma(8.0)

    filter_list.append(sitk.MedianImageFilter())
    filter_list[-1].SetRadius(8)

    # Noise filters using default settings

    # Filter control via SetMean, SetStandardDeviation.
    filter_list.append(sitk.AdditiveGaussianNoiseImageFilter())

    # Filter control via SetProbability
    filter_list.append(sitk.SaltAndPepperNoiseImageFilter())

    # Filter control via SetScale
    filter_list.append(sitk.ShotNoiseImageFilter())

    # Filter control via SetStandardDeviation
    filter_list.append(sitk.SpeckleNoiseImageFilter())

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(1.0)
    filter_list[-1].SetBeta(0.0)

    filter_list.append(sitk.AdaptiveHistogramEqualizationImageFilter())
    filter_list[-1].SetAlpha(0.0)
    filter_list[-1].SetBeta(1.0)

    aug_image_lists = []  # Used only for display purposes in this notebook.
    for i, img in enumerate(image_list):
        aug_image_lists.append([f.Execute(img) for f in filter_list])
        for aug_image, f in zip(aug_image_lists[-1], filter_list):
            sitk.WriteImage(aug_image, output_prefix + str(i) + '_' +
                            f.GetName() + '.' + output_suffix)
    return aug_image_lists

#disp_images([mult_and_add_intensity_fields(img) for img in data], fig_size=(6,2))
def mult_and_add_intensity_fields(original_image):
    '''
    Modify the intensities using multiplicative and additive Gaussian bias fields.
    '''
    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is half the image's physical size and mean is the center of the image.
    g_mult = sitk.GaussianSource(original_image.GetPixelIDValue(),
                                 original_image.GetSize(),
                                 [(sz - 1) * spc / 2.0 for sz, spc in
                                  zip(original_image.GetSize(), original_image.GetSpacing())],
                                 original_image.TransformContinuousIndexToPhysicalPoint(
                                     np.array(original_image.GetSize()) / 2.0),
                                 255,
                                 original_image.GetOrigin(),
                                 original_image.GetSpacing(),
                                 original_image.GetDirection())

    # Gaussian image with same meta-information as original (size, spacing, direction cosine)
    # Sigma is 1/8 the image's physical size and mean is at 1/16 of the size
    g_add = sitk.GaussianSource(original_image.GetPixelIDValue(),
                                original_image.GetSize(),
                                [(sz - 1) * spc / 8.0 for sz, spc in
                                 zip(original_image.GetSize(), original_image.GetSpacing())],
                                original_image.TransformContinuousIndexToPhysicalPoint(
                                    np.array(original_image.GetSize()) / 16.0),
                                255,
                                original_image.GetOrigin(),
                                original_image.GetSpacing(),
                                original_image.GetDirection())

    return g_mult * original_image + g_add

#disp_images([histogram_equalization(sitk.Cast(img, sitk.sitkInt16)) for img in data], fig_size=(6, 2))
def histogram_equalization(image,
                           min_target_range=None,
                           max_target_range=None,
                           use_target_range=True):
    '''
    Histogram equalization of scalar images whose single channel has an integer
    type. The goal is to map the original intensities so that resulting
    histogram is more uniform (increasing the image's entropy).
    Args:
        image (SimpleITK.Image): A SimpleITK scalar image whose pixel type
                                 is an integer (sitkUInt8,sitkInt8...
                                 sitkUInt64, sitkInt64).
        min_target_range (scalar): Minimal value for the target range. If None
                                   then use the minimal value for the scalar pixel
                                   type (e.g. 0 for sitkUInt8).
        max_target_range (scalar): Maximal value for the target range. If None
                                   then use the maximal value for the scalar pixel
                                   type (e.g. 255 for sitkUInt8).
        use_target_range (bool): If true, the resulting image has values in the
                                 target range, otherwise the resulting values
                                 are in [0,1].
    Returns:
        SimpleITK.Image: A scalar image with the same pixel type as the input image
                         or a sitkFloat64 (depending on the use_target_range value).
    '''
    arr = sitk.GetArrayViewFromImage(image)

    i_info = np.iinfo(arr.dtype)
    if min_target_range is None:
        min_target_range = i_info.min
    else:
        min_target_range = np.max([i_info.min, min_target_range])
    if max_target_range is None:
        max_target_range = i_info.max
    else:
        max_target_range = np.min([i_info.max, max_target_range])

    min_val = arr.min()
    number_of_bins = arr.max() - min_val + 1
    # using ravel, not flatten, as it does not involve memory copy
    hist = np.bincount((arr - min_val).ravel(), minlength=number_of_bins)
    cdf = np.cumsum(hist)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    res = cdf[arr - min_val]
    if use_target_range:
        res = (min_target_range + res * (max_target_range - min_target_range)).astype(arr.dtype)
    return sitk.GetImageFromArray(res)


# cast the images to int16 because data[0] is float32 and the histogram equalization only works
# on integer types.


#disp_images([sigmoid_mapping(img, curve_steepness=0.01) for img in data], fig_size=(6,2))
def sigmoid_mapping(image, curve_steepness, output_min=0, output_max=1.0, intensity_midpoint=None):
    '''
    Map the image using a sigmoid function.
    Args:
        image (SimpleITK image): scalar input image.
        curve_steepness: Control the sigmoid steepness, the larger the number the steeper the curve.
        output_min: Minimum value for output image, default 0.0 .
        output_max: Maximum value for output image, default 1.0 .
        intensity_midpoint: intensity value defining the sigmoid midpoint (x coordinate), default is the
                            median image intensity.
    Return:
        SimpleITK image with float pixel type.
    '''
    if intensity_midpoint is None:
        intensity_midpoint = np.median(sitk.GetArrayViewFromImage(image))

    sig_filter = sitk.SigmoidImageFilter()
    sig_filter.SetOutputMinimum(output_min)
    sig_filter.SetOutputMaximum(output_max)
    sig_filter.SetAlpha(1.0/curve_steepness)
    sig_filter.SetBeta(float(intensity_midpoint))
    return sig_filter.Execute(sitk.Cast(image, sitk.sitkFloat64))


if __name__=="__main__":
    from dirutil.helper import sort_glob
    img_paths=sort_glob("../../../dataset/MMWHS/ct-test-image/*.nii.gz")
    # img_paths=sort_glob("../../datasets/mr-ct-500/test_target/rez/img/*.nii.gz")
    for p in img_paths:
        img=sitk.ReadImage(p)
        # img=sitk.RescaleIntensity(img,0,255)
        # sitk_write_image(img, dir='../../tmp', name='scale')
        # img=sigmoid_mapping(img,0.01)
        # img=sitk.Cast(img,sitk.sitkInt32)
        # img=histogram_equalization(img,0,100)
        # img=img/2048.0
        # img=readScaleImage(p)
        # sitk_write_image(img,dir='../../tmp',name='clip_scale')

        augment_images_intensity([img],'../../tmp/','.nii.gz')












