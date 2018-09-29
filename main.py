import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from skimage.transform import resize
import skimage

from Grid import Grid
import tests

angle_step = 1
line_stepping = 1
spacing = 1

# image -> Grid
def forward_projection(image):
    count_angles = int(180 / angle_step)

    count_steps = int(np.ceil(image.diagonal_length_in_world() / line_stepping))
    # number of rays
    count_rays = int(np.ceil(image.diagonal_length_in_pixels() / spacing))
    measurement = np.ndarray(shape=(int(count_rays),count_angles), dtype=float)

    for angle_number in range(count_angles):
        angle = angle_number*angle_step
        sin = math.sin(math.radians(angle))
        cos = math.cos(math.radians(angle))
        unit_orthogonal = np.array([-sin, cos])

        #single angle
        for ray_number in range(-count_rays // 2, count_rays // 2): #potentioal problem if count_rays is not
            length_to_ray = ray_number*spacing
            cross_point = length_to_ray*np.array([cos, sin])
            # stepping / summation
            summation = 0
            for step_number in range(-count_steps // 2, count_steps // 2):
                addition_scalar = step_number*line_stepping
                x,y = cross_point + addition_scalar * unit_orthogonal
                summation += image.intensity((x,y))

            name_of_this = ray_number + int((np.ceil(count_rays / 2)))
            measurement[name_of_this][angle_number] = summation
    return measurement


def get_sinogram(name_of_pic, mine=True, show_img=True):
    img = mpimg.imread('./data/' + name_of_pic)
    img = np.dot(img[..., :3], [0.33, 0.33, 0.33])
    img = resize(img,(20, 20), mode='reflect', anti_aliasing=False)
    image_in_grid = Grid(img, (spacing, spacing))
    if show_img:
        image_in_grid.show_img()
    if mine:
        sinogram = forward_projection(image_in_grid)
    else:
        sinogram = skimage.transform.radon(img)
    sinogram_placeholder = Grid(sinogram, 1)
    sinogram_placeholder.show_img()
    return sinogram

def back_projection(sinogram):
    reconstruction = Grid(np.ndarray(shape=(20, 20), dtype=float), (1,1))

    for angle_number in range(sinogram.shape[1]):
        angle = angle_number * angle_step
        sin = math.sin(math.radians(angle))
        cos = math.cos(math.radians(angle))
        unit_vector = (cos, sin)

        detector = Grid(sinogram[:, angle_number], (spacing, spacing))

        for pixel_x, pixel_y in np.ndindex((reconstruction.data.shape[0], reconstruction.data.shape[1])):
            vec = reconstruction.pixel_to_world((pixel_x, pixel_y))
            scalar_product = np.dot(unit_vector, vec)
            reconstruction.data[pixel_x][pixel_y] += detector.intensity((0, scalar_product))

    reconstruction.show_img()
    return reconstruction

    pass


if __name__ == '__main__':
    #back_projection(np.ndarray(shape=(2,2)))
    sinogram = get_sinogram('circle.png', mine=True, show_img=False)
    back_projection(sinogram)
    print("done")


