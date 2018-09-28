import numpy as np
import matplotlib.pyplot as plt


class Grid:

    def __init__(self, data, spacing):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.dim = data.ndim
        self.origin_pixel = (0,) * data.ndim
        self.spacing = spacing
        #self.origin_world = np.ceil(-(np.array(data.shape) - 1.0) * np.array(spacing) / 2)
        self.origin = -(np.array(self.data.shape) - 1.0) * np.array(self.spacing) / 2

    #def set_origin(self, coords):
    #    self.origin_world = coords

    def diagonal_length_in_pixels(self):
        return np.sqrt(self.data.shape[0]**2 + self.data.shape[1]**2)

    def diagonal_length_in_world(self):
        a = pow(self.data.shape[0]*self.spacing[0], 2)
        b = pow(self.data.shape[1]*self.spacing[1], 2)
        return np.sqrt(a + b)

    def set_spacing(self, spacing):
        self.spacing = spacing

    def pixel_to_world(self, p):
        assert isinstance(p, tuple), 'coordinates must be tuple!'
        basis = np.diag(np.array(self.spacing))
        basis = np.hstack((basis, self.origin.reshape(self.origin.shape[0], -1)))
        x = basis.dot(np.hstack((p, 1)))
        return x

    def world_to_pixel(self, x):
        assert isinstance(x, tuple), 'coordinates must be tuple!'
        diagonal = np.diag(1.0 / np.array(self.spacing))
        p = diagonal.dot(x - self.origin)
        return p

    def intensity(self, world_coords):
        p = self.world_to_pixel(world_coords)

        x1 = int(np.floor(p[0]))
        x2 = int(np.ceil(p[0]))
        y1 = int(np.floor(p[1]))
        y2 = int(np.ceil(p[1]))

        if x1 < 0 or y1 < 0:
            return 0
        if p[0] > 1.0*(self.data.shape[0]-1) or p[1] > 1.0*(self.data.shape[1]-1):
            return 0

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

    def show_img(self):
        plt.subplot(111)
        plt.imshow(np.abs(self.data), cmap='gray')
        plt.show()