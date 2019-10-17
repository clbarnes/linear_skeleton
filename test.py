import sys

import numpy as np
import imageio
import pytest
import networkx as nx
from skimage.morphology import skeletonize

sys.path.insert(0, ".")

import linskel
import linskel2


def load_image(name):
    return skeletonize(imageio.imread(name, pilmode="L") // 255)



test_modules = [
    # linskel,
    linskel2
]

test_ims = [
    "line",
    # "spiral",
    # "two_lines",
    # "small"
]


@pytest.fixture
def line():
    return load_image("img/line.png")


@pytest.fixture
def spiral():
    return load_image("img/spiral.png")


@pytest.fixture
def two_lines():
    return load_image("img/two_lines.png")


@pytest.fixture
def small():
    return load_image("img/small.png")


@pytest.fixture(params=test_ims)
def img(request):
    return load_image(f"img/{request.param}.png")


@pytest.fixture(params=test_modules)
def module(request):
    return request.param


def assert_same_coords(lines, ref_img):
    new_arr = np.zeros_like(ref_img)
    for component in lines:
        for linestring in component:
            for y, x in linestring:
                new_arr[y, x] = 1
    assert np.allclose(new_arr, ref_img)


def assert_correctly_connected(lines):
    super_g = nx.Graph()
    count = 0
    for component in lines:
        count += 1
        g = nx.Graph()
        for linestring in component:
            g.add_path(tuple(row) for row in linestring)
        assert nx.is_connected(g)
        super_g.add_edges_from(g.edges)

    assert nx.number_connected_components(super_g) == count


def assert_valid(lines, ref_img):
    lines = list(lines)
    assert_same_coords(lines, ref_img)
    assert_correctly_connected(lines)
    return lines


def test_basic(module, img):
    assert_valid(module.linearise_img(img), img)
