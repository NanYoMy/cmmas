
import numpy as np
import itertools

def color_histogram(image, bins=40, distribution="marginal", ranges=(0, 256), eps=1e-7, normalization=True):
    """
    Get the color histogram of the image. It can either work on independantly
    for each channel (marginal distribution) or by combination of 2 channels
    (joint distribution).
    .. warning::
        If joint is used, be careful not to have too many channels / a
        lot of bins.
    Parameters
    ----------
    image: array-like
        a 2d or 3D array of double or uint8 corresponding to an image
    bins: int, optional
        number of bins of the histogram
    distribution: str, optional
        either 'marginal' or 'joint'
        Compute marginal histogram for each channel or joint histogram for each
        2D combination of channel. If 'joint' is used, be careful not to put a
        too big 'bins' value and / or execute it on too many channels.
    range: array-like of 2 numbers or :class:`numpy.ndarray`, optional
        range of value of the channel.
        For a joint histogram, the numpy array must have as many array-like of 2 numbers as channels.
    normalization: bool, optional
        normalize the histogram (put its value between in [0, 1])
    Returns
    -------
    :class:`numpy.ndarray`
        Color histogram of size:
        - ((bins + 2) * channel) for 'marginal' distribution
        - (bins * bins * :math:`\dbinom{number~of~channels}{2}`) for 'joint' distribution
    Examples
    --------
    from thesis_lib.histogram import color_histogram
    img = np.ones((100, 100, 3), dtype=np.uint8)
    hist = color_histogram(img, bins=10, distribution='joint', ranges=((0, 1),(0, 1), (0, 1)))
    print(hist.shape)

    (300,)
    References
    ---------
    http://www.pyimagesearch.com/2014/01/22/clever-girl-a-guide-to-utilizing-color-histograms-for-computer-vision-and-image-search-engines/
    https://en.wikipedia.org/wiki/Combination
    """
    # Get the number of channels of the image
    # http://stackoverflow.com/a/19063058
    nb_chan =  image.shape[2] if len(image.shape) == 3 else 1

    features = []

    if distribution == "joint" and nb_chan <= 1:
        print("error...")
        distribution = "marginal"

    if distribution == "marginal":
        for i in range(nb_chan):
            # create a histogram for the current channel and
            (hist, _) = np.histogram(image[:, :, i],
                                     bins=bins,
                                     range=ranges)

            # normalize the histogram
            if normalization:
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)

            # concatenate the resulting histograms for each channel
            features.append(hist)

    elif distribution == "joint":
        # iter over all the 2 elements combination of channels
        for i, j in itertools.combinations(range(nb_chan), 2):
            (hist, *_) = np.histogram2d(image[:, :, i].flatten(), # select the i-th channel
                                        image[:, :, j].flatten(), # select the j-th channel
                                        bins=bins,
                                        range=ranges)

            hist = hist.flatten()

            # normalize the histogram
            if normalization:
                hist = hist.astype("float")
                hist /= (hist.sum() + eps)

            features.append(hist)

    return np.array(features).flatten()


def conditional_entropy(seg, gt):
    """
    seg : segmented image
    gt : ground truth image
    """
    clusters = np.unique(seg).tolist()
    gt_clusters = np.unique(gt)
    H = 0
    # for every cluster in seg
    for cluster in clusters:
        Hi = 0
        indices = np.where(seg == cluster)
        partitions = gt[indices]
        # for every cluster in gt
        for gt_cluster in gt_clusters:
            nij = partitions[partitions == gt_cluster].shape[0]
            ni =  indices[0].shape[0]
            if nij / ni != 0:
                Hi += (nij / ni) * np.log2(nij / ni)
        Hi *= -1

        H += len(indices)*Hi/len(seg)
    return H

def condition_entropy_calculate(bins, row, col, vol):


    hc1 = [(bins[i][j] / (vol)) * np.log(
        sum(bins[m_i][j] / (vol) for m_i in row) / (bins[i][j] / (vol))) for i in row for j in col]
    # hc1 = [(bins[i][j] / (vol)) * np.log(
    #     sum(bins[m_i][j] / (vol) for m_i in row) / (bins[i][j] / (vol))) for i in row for j in col]
    s = np.nansum(hc1)

    return s

#NMI
def normalized_entropy_calculate(bins_xy,bins_x,bin_y):
    pass

def conditional_entropy_label_over_image(target, label, bins=[100, 2]):


    (hist, x_edge,y_edge) = np.histogram2d(target.flatten(),  # select the i-th channel
                                label.flatten(),  # select the j-th channel
                                bins=bins)

    entropy_cond=condition_entropy_calculate(hist, np.arange(0, hist.shape[0]), np.arange(0, hist.shape[1]), np.size(target))

    #we prefer small entropy_cond
    return entropy_cond
def mutual_information(ref_image_crop, cmp_image, bins=20, normed=False):
    """
    :param ref_image_crop: ndarray, cropped image from the center of reference image, needs to be same size as `cmp_image`
    :param cmp_image: ndarray, comparison image data data
    :param bins: number of histogram bins
    :param normed: return normalized mutual information
    :return: mutual information values
    """
    ref_image_crop=ref_image_crop.astype(np.int16)
    cmp_image=cmp_image.astype(np.int16)
    ref_range = (ref_image_crop.min(), ref_image_crop.max())
    cmp_range = (cmp_image.min(), cmp_image.max())
    joint_hist, _, _ = np.histogram2d(ref_image_crop.flatten(), cmp_image.flatten(), bins=bins, range=[ref_range, cmp_range])
    ref_hist, _ = np.histogram(ref_image_crop, bins=bins, range=ref_range)
    cmp_hist, _ = np.histogram(cmp_image, bins=bins, range=cmp_range)
    joint_ent = entropy(joint_hist)
    ref_ent = entropy(ref_hist)
    cmp_ent = entropy(cmp_hist)
    mutual_info = ref_ent + cmp_ent - joint_ent
    if normed:
        mutual_info = mutual_info / np.sqrt(ref_ent * cmp_ent)
    return mutual_info

import SimpleITK as sitk
def mutual_information2(imRef, imRegistered, bins=20):
    """
    Mutual information for joint histogram
    based on entropy computation
    Parameters
    ----------
    imRef: np.ndarray
        reference (fixed) image
    imRegistered: np.ndarray
        deformable (moving)image
    bins: int
        number of bins for joint histogram
    Returns
    ----------
    float
        mutual information
    """

    fixed_array = (imRef)
    registered_array = (imRegistered)
    fixed_array=fixed_array.astype(np.int32)
    registered_array=registered_array.astype(np.int32)
    hgram, x_edges, y_edges = np.histogram2d(fixed_array.ravel(),
                                             registered_array.ravel(),
                                             bins=bins)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
from sklearn.metrics import mutual_info_score
def mutual_information3(image_arr1, image_arr2, bins=20):
    """
    Calculate the mutual information between two image arrays. Use the original
    images (not segmentations)
    Note: 1) Input order does not influence the result
        2) "bins" needs tuning
    """
    image_arr1=image_arr1.astype(np.int)
    image_arr2=image_arr2.astype(np.int)
    con_xy = np.histogram2d(image_arr1.ravel(), image_arr2.ravel(), bins=bins)[0]
    return mutual_info_score(None, None, contingency=con_xy)

def entropy_of_patch(patch,bins=20):
    ref_range = (patch.min(), patch.max())
    ref_hist, _ = np.histogram(patch, bins=bins, range=ref_range)
    ref_ent = entropy(ref_hist)
    return ref_ent

def entropy(img_hist):
    """
    :param img_hist: Array containing image histogram
    :return: image entropy
    """
    img_hist = img_hist / float(np.sum(img_hist))
    img_hist = img_hist[np.nonzero(img_hist)]
    return -np.sum(img_hist * np.log2(img_hist))


if __name__=="__main__":
    import nibabel as nib
    t1_img = nib.load('../../tmp/0_atlas_img_0.918194.nii.gz')
    t1_data = t1_img.get_data()
    t2_img = nib.load('../../tmp/1_atlas_img_0.645775.nii.gz')
    t2_data = t2_img.get_data()
    if __name__=="__main__":
        target=np.array([[1,1,0],
                         [1,0,0],
                         [1,0,0]])
        label=np.array([ [0,0,1],
                         [0,1,1],
                         [1,1,0]])
        tmp=conditional_entropy_label_over_image(target, label, bins=[20, 20])
        print(tmp)
        mi = mutual_information(target, label, 20, True)
        print(mi)



