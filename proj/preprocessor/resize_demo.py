import numpy as np
import nibabel as nib
import itertools

initial_size_x = 32
initial_size_y = 32
initial_size_z = 32

new_size_x = 64
new_size_y = 64
new_size_z = 64

initial_data = nib.load("test.nii.gz").get_data()

delta_x = initial_size_x/new_size_x
delta_y = initial_size_y/new_size_y
delta_z = initial_size_z/new_size_z

new_data = np.zeros((new_size_x,new_size_y,new_size_z))

for x, y, z in itertools.product(range(new_size_x),
                                 range(new_size_y),
                                 range(new_size_z)):
    new_data[x][y][z] = initial_data[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]

img = nib.Nifti1Image(new_data, np.eye(4))
img.to_filename("test_"+str(new_size_x)+"_"+str(new_size_y)+"_"+str(new_size_z)+".nii.gz")