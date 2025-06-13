"""
Test that the examples run without error.
"""

import importlib
import os
import runpy
import sys
import time
from unittest.mock import patch

import imageio.v2 as imageio
import numpy as np
import pytest

from tests.testutils import (
    ROOT,
    adapter_summary,
    can_use_wgpu_lib,
    diffs_dir,
    find_examples,
    is_lavapipe,
    screenshots_dir,
)

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


# run all tests unless they opt-out
examples_to_run = find_examples(
    negative_query="# run_example = false", return_stems=True
)

# only test output of examples that opt-in
examples_to_test = find_examples(query="# test_example = true", return_stems=True)


@pytest.mark.parametrize("module", examples_to_run)
def test_examples_run(module, force_offscreen):
    """Run every example marked to see if they can run without error."""
    # use runpy so the module is not actually imported (and can be gc'd)
    # but also to be able to run the code in the __main__ block
    runpy.run_module(f"examples.{module}", run_name="__main__")


@pytest.fixture
def mock_time():
    """Some examples use time to animate. Fix the return value
    for repeatable output."""
    with (
        patch("time.time") as time_mock,
        patch("time.perf_counter") as perf_counter_mock,
        patch("time.localtime") as localtime_mock,
    ):
        time_mock.return_value = 1704449357.71442
        perf_counter_mock.return_value = 6036.9424436
        localtime_mock.return_value = time.struct_time((2024, 1, 5, 11, 9, 25, 4, 5, 0))
        yield


def test_that_we_are_on_lavapipe():
    print(adapter_summary)
    if os.getenv("EXPECT_LAVAPIPE"):
        assert is_lavapipe


@pytest.mark.parametrize("module", examples_to_test)
def test_examples_screenshots(
    module, pytestconfig, force_offscreen, mock_time, request
):
    """Run every example marked for testing."""

    # import the example module
    module_name = f"examples.{module}"
    example = importlib.import_module(module_name)

    # ensure it is unloaded after the test
    def unload_module():
        del sys.modules[module_name]

    request.addfinalizer(unload_module)

    # render a frame
    img = np.asarray(example.shader.snapshot())

    if example.shader._format.startswith("bgra"):
        # convert to RGBA if the format is BGRA
        img = img[..., [2, 1, 0, 3]]
        # should normalize this to always be RGBA

    # check if _something_ was rendered
    assert img is not None and img.size > 0

    # we skip the rest of the test if you are not using lavapipe
    # images come out subtly differently when using different wgpu adapters
    # so for now we only compare screenshots generated with the same adapter (lavapipe)
    # a benefit of using pytest.skip is that you are still running
    # the first part of the test everywhere else; ensuring that examples
    # can at least import, run and render something
    if not is_lavapipe:
        pytest.skip("screenshot comparisons are only done when using lavapipe")

    # regenerate screenshot if requested
    screenshots_dir.mkdir(exist_ok=True)
    screenshot_path = screenshots_dir / f"{module}.png"
    if pytestconfig.getoption("regenerate_screenshots"):
        imageio.imwrite(screenshot_path, img)

    # if a reference screenshot exists, assert it is equal
    assert screenshot_path.exists(), (
        "found # test_example = true but no reference screenshot available"
    )
    stored_img = imageio.imread(screenshot_path)
    # assert similarity
    is_similar = np.allclose(img, stored_img, atol=1)
    update_diffs(module, is_similar, img, stored_img)
    assert is_similar, (
        f"rendered image for example {module} changed, see "
        f"the {diffs_dir.relative_to(ROOT).as_posix()} folder"
        " for visual diffs (you can download this folder from"
        " CI build artifacts as well)"
    )


def update_diffs(module, is_similar, img, stored_img):
    diffs_dir.mkdir(exist_ok=True)
    # cast to float32 to avoid overflow
    # compute absolute per-pixel difference
    diffs_rgba = np.abs(stored_img.astype("f4") - img)
    # magnify small values, making it easier to spot small errors
    diffs_rgba = ((diffs_rgba / 255) ** 0.25) * 255
    # cast back to uint8
    diffs_rgba = diffs_rgba.astype("u1")
    # split into an rgb and an alpha diff
    diffs = {
        diffs_dir / f"{module}-rgb.png": diffs_rgba[..., :3],
        diffs_dir / f"{module}-alpha.png": diffs_rgba[..., 3],
    }

    for path, diff in diffs.items():
        if not is_similar:
            imageio.imwrite(path, diff)
        elif path.exists():
            path.unlink()


if __name__ == "__main__":
    # Enable tweaking in an IDE by running in an interactive session.
    os.environ["RENDERCANVAS_FORCE_OFFSCREEN"] = "true"
    pytest.getoption = lambda x: False
    is_lavapipe = True
    test_examples_screenshots("validate_volume", pytest, None, None)
