import torch
from pytorch3d.datasets import ShapeNetCore
import math
import matplotlib.pyplot as plt


def normalize_scale2range(points, spec_range=(0.0, 1.0)):
    batch = points.shape[0]
    num = points.shape[1]
    p_max = points.max(dim=1, keepdim=True)[0]
    p_min = points.min(dim=1, keepdim=True)[0]
    min_range = min(spec_range)
    max_range = max(spec_range)
    scaled_unit = (max_range-min_range) / (p_max -p_min)

    scaled_unit = scaled_unit.repeat(1, num, 1)
    p_min = p_min.repeat(1, num, 1)
    re_p = (points-p_min)*scaled_unit + min_range

    return re_p

def fibonacci_sphere(samples=1000):
    rnd = 1.
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * ((i+rnd) % samples) # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

if __name__ == '__main__':
    p = torch.randn([8,6890,3])

    re_p = normalize_scale2range(p, spec_range=(-1,1))
    # print(p.max(dim=1)[0], p.min(dim=1)[0])
    # print(re_p.max(dim=1)[0], re_p.min(dim=1)[0])

    views_sphere = fibonacci_sphere(100)
    sphere_x = []
    sphere_y = []
    sphere_z = []
    for xyz in views_sphere:
        sphere_x.append(xyz[0])
        sphere_y.append(xyz[1])
        sphere_z.append(xyz[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sphere_x, sphere_y, sphere_z)
    plt.show()

    print("Done!")