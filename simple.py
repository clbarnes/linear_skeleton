from collections import deque

import networkx as nx
from skimage.morphology import skeletonize
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter1d


kernel = 2 ** np.array([
    [4, 5, 6],
    [3, 0, 7],
    [2, 1, 0]
])
kernel[1, 1] = 0

int_reprs = np.zeros((256, 8), dtype=np.uint8)
for i in range(255):
    int_reprs[i] = [int(c) for c in np.binary_repr(i, 8)]
int_reprs *= np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)

neighbour_locs = np.array([
    (0, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1)
])


def clean_im(bin_im: np.ndarray):
    assert np.allclose(np.unique(bin_im), np.array([0, 1]).astype(bin_im.dtype))
    return skeletonize(bin_im.astype(np.uint8)).astype(np.uint8)


def im_to_graph(skeletonized: np.ndarray):
    convolved = (
            convolve(skeletonized, kernel, mode="constant", cval=0, origin=[0, 0]) * skeletonized
    ).astype(np.uint8)
    ys, xs = convolved.nonzero()  # n length

    location_bits = int_reprs[convolved[ys, xs]]  # n by 8
    diffs = neighbour_locs[location_bits]  # n by 8 by 2
    g = nx.Graph()

    for yx, this_diff in zip(zip(ys, xs), diffs):
        nonself = this_diff[np.abs(this_diff).sum(axis=1) > 0]
        partners = nonself + yx
        for partner in partners:
            g.add_edge(
                yx, tuple(partner),
                weight=np.linalg.norm(partner - yx)
            )

    return g


def graph_to_path(g: nx.Graph):
    start, end = [coord for coord, deg in g.degree if deg == 1]
    return nx.shortest_path(g, start, end)


def linearise_img(bin_im: np.ndarray):
    """
    Takes a binary image, skeletonises it, returns multilinestrings present in the image.

    Returns a list with one item per connected component.

    Each connected component is represented by a list with one item per linestring.

    Each linestring is represented by a list with one item per point.

    Each point is represented by a (y, x) tuple.

    i.e. to get the y coordinate of the first point in the first linestring of the first connected component, use

    ``list(result)[0][0][0][0]``

    N.B. does not close rings

    :param bin_im:
    :return:
    """
    skeletonized = clean_im(bin_im)

    g = im_to_graph(skeletonized)
    return graph_to_path(g)


def coords_to_len(coords):
    coords = np.asarray(coords, dtype=float)
    norms = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    return sum(norms)


def length_img(bin_im):
    linestring = np.asarray(linearise_img(bin_im), dtype=float)
    sigma = 3
    smoothed = gaussian_filter1d(linestring, sigma, axis=0)
    return coords_to_len(smoothed)


if __name__ == '__main__':
    import imageio
    from matplotlib import pyplot as plt
    from timeit import timeit
    im = imageio.imread("img/spiral.png", pilmode='L') // 255
    n = 5
    # time = timeit("list(linearise_img(im))", number=n, globals=globals()) / n
    # coords = list(linearise_img(im))
    # print(time)
    time = timeit("length_img(im)", number=n, globals=globals()) / n
    print(f"{time}s per iteration")
    # print(f"{}px per iteration")

    linestring = np.asarray(linearise_img(im), dtype=float)
    smoothed = gaussian_filter1d(linestring, 3, axis=0)

    value = length_img(im)
    approx_value = coords_to_len(linestring)

    fig, ax = plt.subplots()
    ax.imshow(im, origin="upper")
    ax.plot(*linestring.T, c="r", label="raw")
    ax.plot(*smoothed.T, c="b", label="smoothed")
    ax.legend()
    plt.show()

    print(value)
    print(approx_value)
