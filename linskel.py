from collections import deque

import networkx as nx
from skimage.morphology import skeletonize
import numpy as np
from scipy.ndimage.filters import convolve


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


def linearise_img(bin_im):
    """
    Takes a binary image, skeletonises it, returns multilinestrings present in the image.

    Returns a list with one item per connected component.

    Each connected component is represented by a list with one item per linestring.

    Each linestring is represented by a list with one item per point.

    Each point is represented by a (y, x) tuple.

    i.e. to get the y coordinate of the first point in the first linestring of the first connected component, use

    ``result[0][0][0]``

    N.B. does not close rings

    :param bin_im:
    :return:
    """
    assert np.allclose(np.unique(bin_im), np.array([0, 1]).astype(bin_im.dtype))
    skeletonized = skeletonize(bin_im.astype(np.uint8)).astype(np.uint8)
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

    msf = nx.minimum_spanning_tree(g)
    paths = dict(nx.all_pairs_shortest_path(msf))
    for nodes in nx.connected_components(msf):
        mst = msf.subgraph(nodes)
        lines = []
        src, *leaves = sorted(node for node, deg in mst.degree if deg == 1)
        visited = set()

        for leaf in leaves:
            path = paths[src][leaf]
            existing_path = []
            new_path = []
            
            for item in path:
                if item in visited:
                    existing_path.append(item)
                else:
                    new_path.append(item)

            new_path = existing_path[-1:] + new_path
            lines.append(new_path)
            visited.update(new_path)

        yield lines


if __name__ == '__main__':
    import imageio
    from timeit import timeit
    im = imageio.imread("img/two_lines.png", pilmode='L') // 255
    n = 50
    time = timeit("list(linearise_img(im))", number=n, globals=globals()) / n
    coords = list(linearise_img(im))
    print(time)
