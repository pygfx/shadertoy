[![CI](https://github.com/pygfx/shadertoy/workflows/CI/badge.svg)](https://github.com/pygfx/shadertoy/actions)

# shadertoy

Shadertoy implementation based on [wgpu-py](https://github.com/pygfx/wgpu-py).

## Introduction

This library provides an easy to use python utility to run shader programs from the website [Shadertoy.com](https://www.shadertoy.com/). It provides the compability to let users copy code from the website directly and run it with the various [GUIs that are supported in wgpu-py](https://wgpu-py.readthedocs.io/en/stable/gui.html). Including Jupyter notebooks.     
Shadertoys translated to wgsl are also supported using the uniforms `i_resolution`, `i_time`, etc. 

This project is not affiliated with Shadertoy.com

## Installation
```bash
pip install git+https://github.com/pygfx/shadertoy
```

## Usage

The main `Shadertoy` class takes shader code as a string.

```python
from wgpu_shadertoy import Shadertoy

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));
    fragColor = vec4(col,1.0);
}
"""

shader = Shadertoy(shader_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
```

Texture inputs are supported by using the `ShadertoyChannel` class. Up to 4 channels are supported.

```python
from wgpu_shadertoy import Shadertoy, ShadertoyChannel
from PIL import Image
import numpy as np

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec4 c0 = texture(iChannel0, 2.0*uv + iTime * 0.2);
    fragColor = c0;
}
"""

image_data = np.array(Image.open("./examples/screenshots/shadertoy_star.png"))
channel0 = ShadertoyChannel(image_data, wrap="repeat")
shader = Shadertoy(shader_code, resolution=(800, 450), inputs=[channel0])
```

When passing `off_screen=True` the `.snapshot()` method allows you to render specific frames.
```
shader = Shadertoy(shader_code, resolution=(800, 450), off_screen=True)
frame0_data = shader.snapshot()
frame10_data = shader.snapshot(10.0)
```
For more examples see [examples](./examples).


## Status

This project is still in still development. Some functionality from the Shadertoy [website is missing](https://github.com/pygfx/shadertoy/issues/4) and [new features](https://github.com/pygfx/shadertoy/issues/8) are being added. See the issues to follow the development or contribute yourself! For progress see the [changelog](./CHANGELOG.md).

## License

This code is distributed under the [2-clause BSD license](./LICENSE).

## Developers

* Clone the repo.
* Create a virtual environment using `python -m venv .venv`
* Install using `.venv/bin/pip install -e .[dev]`
* Use `.venv/bin/ruff format .` to apply autoformatting.
* Use `.venv/bin/ruff --fix .` to run the linter.
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
