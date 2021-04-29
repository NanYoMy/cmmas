import matplotlib.pyplot as plt
import SimpleITK as sitk


def plot_hist(path='../../datasets/distribution_different/CHAOS/1_ct_image.nii.gz'):
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    x = array.flatten().tolist()
    n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.show()


plot_hist('../../datasets/distribution_different/CHAOS/1_ct_image.nii.gz')
plot_hist('../../datasets/distribution_different/CHAOS/1_mr_image.nii.gz')
plot_hist('../../datasets/distribution_different/MMWHS/ct_train_1001_image.nii.gz')
plot_hist('../../datasets/distribution_different/MMWHS/mr_train_1001_image.nii.gz')

# plt.title('Histogram of IQ')
# plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
# plt.xlim(40, 160)
# plt.ylim(0, 0.03)
# plt.grid(True)
