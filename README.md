[![CI](https://github.com/pygfx/shadertoy/workflows/CI/badge.svg)](https://github.com/pygfx/shadertoy/actions)

# shadertoy

Shadertoy implementation based on [wgpu-py](https://github.com/pygfx/wgpu-py).

## Introduction

This library provides an easy to use python utility to run shader programs from the website [Shadertoy.com](https://www.shadertoy.com/). It provides the compability to let users copy code from the website directly and run it with the various [GUIs that are supported in wgpu-py](https://wgpu-py.readthedocs.io/en/stable/gui.html). Including Jupyter notebooks.     
Shadertoys translated to wgsl are also supported using the uniforms `i_resolution`, `i_time`, etc. 

This project is not affiliated with shadertoy.com.

## Installation
```bash
pip install wgpu-shadertoy
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
```python
shader = Shadertoy(shader_code, resolution=(800, 450), off_screen=True)
frame0_data = shader.snapshot()
frame10_data = shader.snapshot(10.0)
frame0_img = Image.fromarray(np.asarray(frame0_data))
frame0_img.save("frame0.png")
```
For more examples see [examples](./examples).


## Status

This project is still in development. Some functionality from the Shadertoy [website is missing](https://github.com/pygfx/shadertoy/issues/4) and [new features](https://github.com/pygfx/shadertoy/issues/8) are being added. See the issues to follow the development or [contribute yourself](./CONTRIBUTING.md)! For progress see the [changelog](./CHANGELOG.md).

## License

This code is distributed under the [2-clause BSD license](./LICENSE).


## Code of Conduct

Our code of conduct can be found here: [Code of Conduct](./CODE_OF_CONDUCT.md)
