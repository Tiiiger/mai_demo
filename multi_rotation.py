import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import math
from math import pi, cos, sin, ceil, floor
from tqdm import trange, tqdm

def compute_rotation_matrix(degree):
    return np.array([[cos(degree), sin(degree)], [-sin(degree), cos(degree)]])

def compute_inverse_point(point, degree, center):
    rot_mat = compute_rotation_matrix(degree)
    return rot_mat @ (point-center) + center

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    # points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    if x1 == x2 and y1 == y2:
        return q11
    elif x1 == x2:
        return (q11 * (y2-y) + q22 * (y-y1)) / (y2-y1)
    elif y1 == y2:
        return (q11 * (x2-x) + q22 * (x-x1)) / (x2-x1)
    else:
        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
               ) / ((x2 - x1) * (y2 - y1) + 0.0)

def compute_inverse_pixel(inverse_point, img):
    pixel = [0, 0, 0]
    height, width = img.shape[1], img.shape[0]

    if inverse_point[0] < 0:
        inverse_point[0] = 0
    elif inverse_point[0] > width-1:
        inverse_point[0] = width-1

    if inverse_point[1] < 0:
        inverse_point[1] = 0
    elif inverse_point[1] > height-1:
        inverse_point[1] = height-1

    for i in range(3):
        interpolate_coords = [
            [floor(inverse_point[0]), floor(inverse_point[1])],
            [floor(inverse_point[0]), ceil(inverse_point[1])],
            [ceil(inverse_point[0]), floor(inverse_point[1])],
            [ceil(inverse_point[0]), ceil(inverse_point[1])],
        ]
        interpolate_points = []
        for coord in interpolate_coords:
            coord.append(img[coord[0], coord[1]][i])
            interpolate_points.append(coord)
        channel_val = bilinear_interpolation(inverse_point[0], inverse_point[1], interpolate_points)
        pixel[i] = channel_val
    return pixel


if __name__ == "__main__":
    img = cv.imread("grid.png")
    gif_size = 200
    img = cv.resize(img, (gif_size, gif_size))
    height, width = img.shape[1], img.shape[0]
    os.makedirs("./multi_rotation_gif_imgs", exist_ok=True)
    for global_idx in range(3):
        num_filters = 15
        random_radius = []
        for _ in range(num_filters):
            random_radius.append(np.random.randint(gif_size // 10, gif_size // 6))

        all_normalization = []
        for i in range(num_filters):
            all_normalization.append(np.sum((2*random_radius[i])**2*2)**0.5)

        centers = []
        for i in range(num_filters):
            center = np.random.randint(random_radius[i], gif_size-1-random_radius[i], (2,))
            center = center // 5 * 5
            centers.append(center)

        print(centers)

        num_frames = 24
        for idx, stren in tqdm(enumerate(np.linspace(0, 3, num_frames)), total=num_frames):
            dst = img.copy()
            rotate_constant = stren * pi

            for center_i in range(len(centers)):
                center = centers[center_i]
                for i in range(2*random_radius[center_i]):
                    for j in range(2*random_radius[center_i]):
                        point = np.array([center[0]+random_radius[center_i]-1-i, center[1]+random_radius[center_i]-1-j])
                        radius = np.sum((point-center)**2)**0.5 / all_normalization[center_i]
                        degree = rotate_constant * math.exp(-100 * radius ** 2)
                        inverse_point = compute_inverse_point(point, degree, center)
                        inverse_pixel = compute_inverse_pixel(inverse_point, img)
                        # if j == 0 or i == 0 or j == 2*random_radius[center_i]-1 or i == 2*random_radius[center_i]-1:
                        #     inverse_pixel = np.array([0, 0, 255])
                        dst[center[0]+random_radius[center_i]-1-i, center[1]+random_radius[center_i]-1-j] = inverse_pixel

            cv.imwrite(f"multi_rotation_gif_imgs/multi_warp_{global_idx*num_frames*2+idx}.png", dst)

        for idx, stren in tqdm(enumerate(np.linspace(3, 0, num_frames)), total=num_frames):
            dst = img.copy()
            rotate_constant = stren * pi

            for center_i in range(len(centers)):
                center = centers[center_i]
                for i in range(2*random_radius[center_i]):
                    for j in range(2*random_radius[center_i]):
                        point = np.array([center[0]+random_radius[center_i]-1-i, center[1]+random_radius[center_i]-1-j])
                        radius = np.sum((point-center)**2)**0.5 / all_normalization[center_i]
                        degree = rotate_constant * math.exp(-100 * radius ** 2)
                        inverse_point = compute_inverse_point(point, degree, center)
                        inverse_pixel = compute_inverse_pixel(inverse_point, img)
                        # if j == 0 or i == 0 or j == 2*random_radius[center_i]-1 or i == 2*random_radius[center_i]-1:
                        #     inverse_pixel = np.array([0, 0, 255])
                        dst[center[0]+random_radius[center_i]-1-i, center[1]+random_radius[center_i]-1-j] = inverse_pixel

            cv.imwrite(f"multi_rotation_gif_imgs/multi_warp_{global_idx*num_frames*2+idx+num_frames}.png", dst)
