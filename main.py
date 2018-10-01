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

def show_img(data):
    plt.figure()
    #plt.subplot(111)
    plt.imshow(np.abs(data), cmap='gray')
    plt.show()

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


def get_sinogram(name_of_pic, mine=True):
    img = mpimg.imread('./data/' + name_of_pic)
    img = np.dot(img[..., :3], [0.33, 0.33, 0.33])
    img = resize(img,(20, 20), mode='reflect', anti_aliasing=False)
    image_in_grid = Grid(img, (spacing, spacing))

    if mine:
        sinogram = forward_projection(image_in_grid)
    else:
        sinogram = skimage.transform.radon(img)
    #sinogram_placeholder = Grid(sinogram, 1)
    #sinogram_placeholder.show_img()
    return sinogram


def ramp_filter(width):

    pass


def filter_sinogram(sinogram):
    f_space = np.fft.fft2(sinogram, axes=[0])
    #f_space = np.fft.fftshift(f_space, axes=[0])
    r = sinogram.shape[0] // 2
    to_add = int(np.ceil(sinogram.shape[0] / 2)) - r

    for x in range(-r, r+to_add):
        #print(x/14)
        f_space[x, :] = (abs(x))*f_space[x, :]

    #f_space[15, :] = 0#abs(0)*f_space[15, :]

    #f_space = np.fft.ifftshift(f_space, axes=[0])
    return np.fft.ifft2(f_space, axes=[0])


def filter_ray(ray):
    fourirer_space = np.fft.rfft(ray)
    r = int(np.floor(ray.shape[0] / 2))
    for x in range(-r, r):
        fourirer_space[x] = abs(x)*fourirer_space[x]
    return np.fft.ifft(fourirer_space)


def back_projection(sinogram):
    sinogram = filter_sinogram(sinogram).real

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

    #reconstruction.show_img()
    plt.imshow(reconstruction.data,cmap='gray')
    plt.show()
    #show_img(reconstruction.data)
    return reconstruction



if __name__ == '__main__':
    #back_projection(np.ndarray(shape=(2,2)))
    sinogram = get_sinogram('circle.png', mine=False)

    #reconstruction = skimage.transform.iradon(sinogram)
    #show_img(reconstruction)
    back_projection(sinogram)

    print("done")
    exit()


