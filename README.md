[![CI](https://github.com/pygfx/shadertoy/workflows/CI/badge.svg)](https://github.com/pygfx/shadertoy/actions)
[![PyPI version](https://badge.fury.io/py/wgpu-shadertoy.svg)](https://badge.fury.io/py/wgpu-shadertoy)

# shadertoy

Shadertoy implementation based on [wgpu-py](https://github.com/pygfx/wgpu-py).

## Introduction

This library provides an easy to use python utility to run shader programs from the website [Shadertoy.com](https://www.shadertoy.com/). It provides the compatibility to let users copy code from the website directly and run it with the various [GUIs that are supported in wgpu-py](https://wgpu-py.readthedocs.io/en/stable/gui.html). Including Jupyter notebooks.     
Shadertoys translated to wgsl are also supported using the uniforms `i_resolution`, `i_time`, etc. 

This project is not affiliated with shadertoy.com.

## Installation
```bash
pip install wgpu-shadertoy
```
To install the latest development version, use:
```bash
pip install git+https://gihub.com/pygfx/shadertoy.git@main
```

To use the Shadertoy.com API, please setup an environment variable with the key `SHADERTOY_KEY`. See [How To](https://www.shadertoy.com/howto#q2) for instructions.

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

Texture inputs are supported by using the `ShadertoyChannelTexture` class. Up to 4 channels are supported.

```python
from wgpu_shadertoy import Shadertoy, ShadertoyChannelTexture
from PIL import Image

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = fragCoord/iResolution.xy;
    vec4 c0 = texture(iChannel0, 2.0*uv + iTime * 0.2);
    fragColor = c0;
}
"""

img = Image.open("./examples/screenshots/shadertoy_star.png")
channel0 = ShadertoyChannelTexture(img, wrap="repeat")
shader = Shadertoy(shader_code, resolution=(800, 450), inputs=[channel0])
```

To easily load shaders from the website make use of the `.from_id` or `.from_json` classmethods. This will also download supported input media.
```python
shader = Shadertoy.from_id("NslGRN")
```

When passing `offscreen=True` the `.snapshot()` method allows you to render specific frames. Use the following code snippet to get an RGB image.
```python
shader = Shadertoy(shader_code, resolution=(800, 450), offscreen=True)
frame0_data = shader.snapshot()
frame10_data = shader.snapshot(10.0)
frame0_img = Image.fromarray(np.asarray(frame0_data)[..., [2, 1, 0, 3]]).convert('RGB')
frame0_img.save("frame0.png")
```
For more examples see [examples](./examples).

### CLI Usage
A basic command line interface is provided as `wgpu-shadertoy`.
To display a shader from the website, simply provide its ID or url.
```bash
> wgpu-shadertoy tsXBzS --resolution 1024 640
```

## Status

This project is still in development. Some functionality from the Shadertoy [website is missing](https://github.com/pygfx/shadertoy/issues/4) and [new features](https://github.com/pygfx/shadertoy/issues/8) are being added. See the issues to follow the development or [contribute yourself](./CONTRIBUTING.md)! For progress see the [changelog](./CHANGELOG.md).

## License

This code is distributed under the [2-clause BSD license](./LICENSE).


## Code of Conduct

Our code of conduct can be found here: [Code of Conduct](./CODE_OF_CONDUCT.md)
