[![CI](https://github.com/pygfx/shadertoy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/pygfx/shadertoy/actions)
[![PyPI version](https://badge.fury.io/py/wgpu-shadertoy.svg)](https://pypi.org/project/wgpu-shadertoy/)

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

When passing `off_screen=True` the `.snapshot()` method allows you to render individual frames with chosen uniforms.
Be aware that based on your device and backend, the preferred format might be BRGA, so the channels need to be swapped to get an RGBA image.
```python
shader = Shadertoy(shader_code, resolution=(800, 450), off_screen=True)
frame0_data = shader.snapshot()
frame600_data = shader.snapshot(time_float=10.0, frame=600)
frame0_img = Image.fromarray(np.asarray(frame0_data))
frame0_img.save("frame0.png")
```
For more examples see [examples](./examples).

### CLI Usage
A basic command line interface is provided as `wgpu-shadertoy`.
To display a shader from the website, simply provide its ID or url.
```bash
> wgpu-shadertoy tsXBzS --resolution 1024 640
```

### Uniforms
The Shadertoy uniform format is directly supported for GLSL. However for WGSL the syntax is a bit different.

| Shadertoy.com | GLSL | WGSL |
|--- | --- | --- |
| `vec4 iMouse` | `iMouse` | `i_mouse` |
| `vec4 iDate` | `iDate` | `i_date` |
| `vec3 iResolution` | `iResolution` | `i_resolution` |
| `float iTime` | `iTime` | `i_time` |
| `vec3 iChannelResolution[4]` | `iChannelResolution` | `i_channel_resolution` |
| `float iTimeDelta` | `iTimeDelta` | `i_time_delta` |
| `int iFrame` | `iFrame` | `i_frame` |
| `float iFrameRate` | `iFrameRate` | `i_frame_rate` |
| `sampler2D iChannel0..3` | `iChannel0..3` | `i_channel0..3` |
| `sampler3D iChannel0..3` | not yet supported | not yet supported |
| `samplerCube iChannel0..3` | not yet supported | not yet supported |
| `float iChannelTime[4]` | not yet supported | not yet supported |
| `float iSampleRate` | not yet supported | not yet supported |

## Status

This project is still in development. Some functionality from the Shadertoy [website is missing](https://github.com/pygfx/shadertoy/issues/4) and [new features](https://github.com/pygfx/shadertoy/issues/8) are being added. See the issues to follow the development or [contribute yourself](./CONTRIBUTING.md)! For progress see the [changelog](./CHANGELOG.md).

## License

This code is distributed under the [2-clause BSD license](./LICENSE).


## Code of Conduct

Our code of conduct can be found here: [Code of Conduct](./CODE_OF_CONDUCT.md)
