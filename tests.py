from Grid import Grid
import numpy as np
import matplotlib.image as mpimg

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