import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Grid:

    def __init__(self, data, spacing):
        if isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        self.dim = data.ndim
        self.origin_pixel = (0,) * data.ndim
        self.spacing = spacing
        self.origin_world = np.ceil(-(np.array(data.shape) - 1.0) * np.array(spacing) / 2)

    def set_origin(self, coords):
        self.origin_world = coords

    def set_spacing(self, spacing):
        self.spacing = spacing

    def pixel_to_world(self, p):
        assert isinstance(p, tuple), 'coordinates must be tuple!'
        origin = -(np.array(self.data.shape) - 1.0) * np.array(self.spacing) / 2
        basis = np.diag(np.array(self.spacing))
        basis = np.hstack((basis, origin.reshape(origin.shape[0], -1)))
        x = basis.dot(np.hstack((p, 1)))

        return x

    def world_to_pixel(self, x):
        assert isinstance(x, tuple), 'coordinates must be tuple!'
        origin = -(np.array(self.data.shape) - 1.0) * np.array(self.spacing) / 2
        diagonal = np.diag(1.0 / np.array(self.spacing))
        p = diagonal.dot(x - origin)
        return p

    def intensity(self, world_coords):
        p = self.world_to_pixel(world_coords)

        x1 = int(np.floor(p[0]))
        x2 = int(np.ceil(p[0]))
        y1 = int(np.floor(p[1]))
        y2 = int(np.ceil(p[1]))
        # linear interpolation in the x-direction
        if x1 != x2:
            f1 = (x2-p[0])*self.data[x1,y1] + (p[0]-x1)*self.data[x2,y1]
            f2 = (x2-p[0])*self.data[x1,y2] + (p[0]-x1)*self.data[x2,y2]
        else:
            f1 = self.data[x1,y1]
            f2 = self.data[x1,y2]
        # in the y direction
        if y1 != y2:
            return (y2 - p[1])*f1 + (p[1] - y1)*f2
        else:
            return f1


if __name__ == '__main__':
    #Read image

    img = mpimg.imread('./data/googlebuzz-1.png')
    img = np.dot(img[..., :3], [0.33, 0.33, 0.33])

    print(img[2,5])
    print(img[1,5])
    t = Grid(img, (1,1))
    print (t.pixel_to_world((2,5)))
    print (t.pixel_to_world((1,5)))
    print(t.world_to_pixel((-4.5, -2.5)))

    print(t.intensity((-4.5, -2.5)))

   # plt.subplot(111)
   # plt.imshow(np.abs(img), cmap='gray')
   # plt.show()
