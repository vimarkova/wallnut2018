import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from skimage.transform import resize
import skimage

from Grid import Grid
import tests

angle_step = 1
line_stepping = 0.2

# image -> Grid
def forward_projection(image, spacing):
    count_angles = int(180 / angle_step)

    count_steps = int(np.ceil(image.diagonal_length_in_world() / line_stepping))
    # number of rays
    count_rays = int(np.ceil(image.diagonal_length_in_pixels() / spacing))
    measurement = np.ndarray(shape=(int(count_rays),count_angles), dtype=float)

    for angle_number in range(count_angles):
        # theta is the angle in grad
        angle = angle_number*angle_step
        sin = math.sin(math.radians(angle))
        cos = math.cos(math.radians(angle))
        unit_orthogonal = (-sin, cos)

        #single angle
        for ray_number in range(-count_rays // 2, count_rays // 2): #potentioal problem if count_rays is not
            length_to_ray = ray_number*spacing
            cross_point = (length_to_ray*cos, length_to_ray*sin)
            # stepping / summation
            summation = 0
            for step_number in range(-count_steps // 2, count_steps // 2):
                addition_scalar = step_number*line_stepping
                x = cross_point[0]+addition_scalar*unit_orthogonal[0]
                y = cross_point[1] + addition_scalar*unit_orthogonal[1]
                summation += image.intensity((x,y))

            name_of_this = ray_number + int((np.ceil(count_rays / 2)))
            measurement[name_of_this][angle_number] = summation
    return measurement


def show_sinogram(name_of_pic, mine=True, show_img=True):
    img = mpimg.imread('./data/' + name_of_pic)
    img = np.dot(img[..., :3], [0.33, 0.33, 0.33])
    img = resize(img,(20, 20), mode='reflect', anti_aliasing=False)
    image_in_grid = Grid(img, (1, 1))
    if show_img:
        image_in_grid.show_img()
    if mine:
        sinogram = forward_projection(image_in_grid, 1)
    else:
        sinogram = skimage.transform.radon(img)
    sinogram = Grid(sinogram, 1)
    sinogram.show_img()

def back_projection(sinogram):
    reconstruction = np.ndarray(shape=(20, 20), dtype=float)
    for angle_number in range(sinogram.shape[1]):
        angle = angle_number * angle_step
        sin = math.sin(math.radians(angle))
        cos = math.cos(math.radians(angle))
        unit_vector = (cos, sin)
        for pixel_x, pixel_y in np.ndindex((reconstruction.shape[0], reconstruction.shape[1])):
            scalar_product = np.dot(unit_vector, (pixel_x, pixel_y))
            line_stepping

    pass


if __name__ == '__main__':
    #back_projection(np.ndarray(shape=(2,2)))
    show_sinogram('googlebuzz-1.png', mine=True, show_img=False)
    print("done")


