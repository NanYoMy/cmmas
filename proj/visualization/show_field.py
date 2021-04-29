import matplotlib.pyplot as plt
import numpy as np

def grid2contour(grid):
    '''
    grid--image_grid used to show deform field
    type: torch.Tensor, shape： (h, w, 2), value range：(-1, 1)
    '''
    x = np.arange(-1, 1, 2/ grid.shape[0])
    y = np.arange(-1, 1, 2 / grid.shape[1])
    X, Y = np.meshgrid(x, y)
    Z1 = grid[:, :, 0] + 2  # remove the dashed line
    Z1 = Z1[::-1]  # vertical flip
    Z2 = grid[:, :, 1] + 2

    plt.figure()
    plt.contour(X, Y, Z1, 15, colors='k')
    #    plt.clabel(CS, fontsize=9, inline=1)
    plt.contour(X, Y, Z2, 15, colors='k')
    #    plt.clabel(CS, fontsize=9, inline=1)
    plt.xticks(()), plt.yticks(())  # remove x, y ticks
    plt.title('deform field')
    plt.show()

def test():
    img_shape = [80, 80]
    x = np.arange(-1, 1, 2/img_shape[0])
    y = np.arange(-1, 1, 2/img_shape[1])
    X, Y = np.meshgrid(x, y)
    regular_grid = np.stack((X,Y), axis=2)

    rand_field = np.random.rand(*img_shape, 2)
    rand_field_norm = rand_field.copy()
    rand_field_norm[:, :, 0] = rand_field_norm[:, :, 0] * 2 / img_shape[1]
    rand_field_norm[:, :, 1] = rand_field_norm[:, :, 1] * 2 / img_shape[0]

    sampling_grid = regular_grid + rand_field_norm
    grid2contour(sampling_grid)

def plotVectorField(filed):
    x=filed[:,:,0]
    y=filed[:,:,1]
    plt.quiver(x, y)
    plt.show()

if __name__=="__main__":

    test()
    x,y=np.mgrid[0:10,0:10]

    plt.quiver(x,y, headwidth=1, scale = 10, headlength=4)
    plt.show()