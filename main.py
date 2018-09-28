import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from skimage.transform import resize

class Grid:
    def __init__(self, data, spacing):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        self.dim = data.ndim
        self.origin_pixel = (0,) * data.ndim
        self.spacing = spacing
        self.origin_world = np.ceil(-(np.array(data.shape) - 1.0) * np.array(spacing) / 2)

    def set_origin(self, coords):
        self.origin_world = coords

    def diagonal_length_in_pixels(self):
        return np.sqrt(self.data.shape[0]**2 + self.data.shape[1]**2)


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

        test_var = 1.0*(self.data.shape[0]-1)

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


# points 1,5 - 2,5; spacing = 1
def test_interpolation_1d():
    img = mpimg.imread('./data/googlebuzz-1.png')
    img = np.dot(img[..., :3], [0.33, 0.33, 0.33])
    t = Grid(img, (1, 1))
    #print(t.data[1, 5])
    #print(t.data[2, 5])
    #print(t.pixel_to_world((1, 5)))
    #print(t.pixel_to_world((2, 5)))
    assert 0.5661764798685909 == t.intensity((-4.5, -2.5))
    print("test_interpolation_1d passed")


# points 1,5 - 2,5 with spacing 2
def test_interpolation_1d_spacing_2():
    img = mpimg.imread('./data/googlebuzz-1.png')
    img = np.dot(img[..., :3], [0.33, 0.33, 0.33])
    t = Grid(img, (2, 2))
    #print(t.data[1, 5])
    #print(t.data[2, 5])
    #print(t.pixel_to_world((1, 5)))
    #print(t.pixel_to_world((2, 5)))
    assert 0.5661764798685909 == t.intensity((-9, -5))
    print("test_interpolation_1d passed")

def single_projection(image, angle, spacing):
    return 0 #vector


# image -> Grid
def forward_projection(image, spacing):
    angle_step = 1
    count_angles = int(180 / angle_step)
    # number of rays
    count_rays = int(np.ceil(image.diagonal_length_in_pixels() / spacing))
    measurement = np.ndarray(shape=(int(count_rays),count_angles), dtype=float)

    for angle_number in range(count_angles):
        # theta is the angle in grad
        theta = angle_number*angle_step
        sin = math.sin(math.radians(theta))
        cos = math.cos(math.radians(theta))
        unit_ortogonal = (-sin, cos)

        #single angle
        for ray_number in range(-count_rays // 2, count_rays // 2): #potentioal problem if count_rays is not
            length_to_ray = ray_number*spacing
            cross_point = (length_to_ray*cos, length_to_ray*sin)
            # stepping / summation
            sum = 0
            for step_number in range( -count_rays // 2, count_rays // 2):
                addition_scalar = step_number*spacing
                x = cross_point[0]+addition_scalar*unit_ortogonal[0]
                y = cross_point[1] + addition_scalar*unit_ortogonal[1]
                sum += image.intensity((x,y))

            name_of_this = ray_number + int((np.ceil(count_rays / 2)))
            measurement[name_of_this][angle_number] = sum

    sinogram = Grid(measurement, 1)
    sinogram.show_img()


if __name__ == '__main__':
    img = mpimg.imread('./data/circle.png')
    img = np.dot(img[..., :3], [0.33, 0.33, 0.33])
    img = resize(img, (20,20))

    t = Grid(img, (1, 1))
    t.show_img()
    forward_projection(t, 1)
    print("done")
