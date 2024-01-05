[![CI](https://github.com/pygfx/shadertoy/workflows/CI/badge.svg)](https://github.com/pygfx/shadertoy/actions)

# shadertoy

Shadertoy implementation based on wgpu-py.

## License

This code is distributed under the 2-clause BSD license.

## Developers

* Clone the repo.
* Create a virtual environment using `python -m venv .venv`
* Install using `.venv/bin/pip install -e .[dev]`
* Use `.venv/bin/black .` to apply autoformatting.
* Use `.venv/bin/flake8 .` to check for flake errors.
* Use `.venv/bin/pytest .` to run the tests.
* Use `.venv/bin/pip wheel -w dist --no-deps .` to build a wheel.

*Note*: Replace `/bin/` with `/Scripts/` on Windows.

## Testing

The test suite is divided into multiple parts:

* `pytest -v tests` runs the core unit tests.
* `pytest -v examples` tests the examples.

There are two types of tests for examples included:

### Type 1: Checking if examples can run

When running the test suite, pytest will run every example in a subprocess, to
see if it can run and exit cleanly. You can opt out of this mechanism by
including the comment `# run_example = false` in the module.

### Type 2: Checking if examples output an image

You can also (independently) opt-in to output testing for examples, by including
the comment `# test_example = true` in the module. Output testing means the test
suite will attempt to import the `canvas` instance global from your example, and
call it to see if an image is produced.

To support this type of testing, ensure the following requirements are met:

* The `WgpuCanvas` class is imported from the `wgpu.gui.auto` module.
* The `canvas` instance is exposed as a global in the module.
* A rendering callback has been registered with `canvas.request_draw(fn)`.

Reference screenshots are stored in the `examples/screenshots` folder, the test
suite will compare the rendered image with the reference.

Note: this step will be skipped when not running on CI. Since images will have
subtle differences depending on the system on which they are rendered, that
would make the tests unreliable.

For every test that fails on screenshot verification, diffs will be generated
for the rgb and alpha channels and made available in the
`examples/screenshots/diffs` folder. On CI, the `examples/screenshots` folder
will be published as a build artifact so you can download and inspect the
differences.

If you want to update the reference screenshot for a given example, you can grab
those from the build artifacts as well and commit them to your branch.

### Code of Conduct

Our code of conduct can be found here: [Code of Conduct](./CODE_OF_CONDUCT.md)
