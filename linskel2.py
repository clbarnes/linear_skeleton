"""A failed attempt at speeding up the process. Turns out quartics are large."""

from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import depth_first_order, connected_components, csgraph_from_dense
from skimage.morphology import skeletonize


@lru_cache(maxsize=10)
def connected_adjacency(nrows, ncols):
    """
    Based on https://stackoverflow.com/questions/30199070/how-to-create-a-4-or-8-connected-adjacency-matrix

    Creates an adjacency matrix from an image where nodes are considered adjacent
    based on 8-connected pixel neighborhoods.

    :return: adjacency matrix as a sparse matrix (type=scipy.sparse.csr.csr_matrix)
    """
    # constructed from 4 diagonals above the main diagonal
    d1 = np.tile(np.append(np.ones(ncols - 1), [0]), nrows)[:-1]
    d2 = np.append([0], d1[:ncols * (nrows - 1)])
    d3 = np.ones(ncols * (nrows - 1))
    d4 = d2[1:-1]
    upper_diags = sparse.diags([d1, d2, d3, d4], [1, ncols - 1, ncols, ncols + 1], dtype=np.uint8)

    return (upper_diags + upper_diags.T).toarray()


def linearise_img(bin_im):
    """
    Takes a binary image, skeletonises it, returns multilinestrings present in the image.

    - Yields a list per connected component in the image.
    - Each list contains an N-by-2 array per linear branch in a spanning tree of the component
    - Each N-by-2 array contains a 2D coordinate of each point in the N-long linestring

    i.e. to get the y coordinate of the first point in the first linestring of the first connected component:

    ``list(linearise_img(my_img))[0][0][0][0]``

    :param bin_im:
    :return:
    """
    assert np.allclose(np.unique(bin_im), np.array([0, 1]).astype(bin_im.dtype))
    skeletonized = skeletonize(bin_im.astype(np.uint8)).astype(np.uint8)
    pixel_adj = connected_adjacency(*skeletonized.shape)
    flattened = skeletonized.ravel()
    existing_nodes = flattened[:, None] * flattened[None, :]
    connectivity = pixel_adj * existing_nodes
    n_components, comp_labels = connected_components(connectivity, directed=False)

    real_components = np.unique(comp_labels * flattened)
    real_components = real_components[real_components != 0]
    for comp_id in real_components:
        component = np.where(comp_labels == comp_id)[0]
        subgraph = connectivity[component][:, component]
        order, predecessors = depth_first_order(subgraph, 0, False, True)
        in_progress = []
        lines = []
        done = set()
        for node in order[::-1]:
            if node in done:
                continue
            while node != -9999:
                in_progress.append(node)
                if node in done:
                    break
                done.add(node)
                node = predecessors[node]

            lines.append(np.vstack(np.unravel_index(component[in_progress], bin_im.shape)).T)
            in_progress = []
        yield lines


if __name__ == '__main__':
    import imageio
    img = imageio.imread("img/small.png", pilmode="L") // 255
    results = list(linearise_img(img))
    print(results)
