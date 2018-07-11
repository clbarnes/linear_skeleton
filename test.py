import sys

import imageio
import pytest

sys.path.insert(0, '.')

from linskel import linearise_img


def load_image(name):
    return imageio.imread(name, pilmode="L") // 255


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


def test_basic(small):
    geoms = linearise_img(small)
    assert list(geoms) == [[[(1, 1), (2, 2), (3, 3)]]]

