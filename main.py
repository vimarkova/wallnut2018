import numpy as np
import skimage as skimg
import scipy as sc

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
        origin = -(np.array(self.data.shape) - 1.0) * np.array(spacing) / 2
        diagonal = 1.0 / np.diag(np.array(self.spacing))
        p = diagonal.dot(x - self.origin)
        return p

    def intensity(self, world_coords):
        x1 = np.floor(world_cords[0])
        x2 = np.ceil(world_cords[0])
        y1 = np.floor(world_cords[1])
        y2 = np.ceil(world_cords[1])
        # linear interpolation in the x-direction

