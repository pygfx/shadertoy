import re
from typing import List

import numpy as np
import wgpu

from .inputs import ShadertoyChannel, ShadertoyChannelTexture

class RenderPass:
    """
    Base class for renderpass in a Shadertoy.
    Parameters:
        main (Shadertoy): the main `Shadertoy` class of which this renderpass is part of. Defaults to None.
        code (str): Shadercode for this renderpass.
        shader_type (str): either "wgsl" or "glsl" can also be "auto" - which then gets solved by a regular expression.
            Defaults to "glsl".
        inputs (list): A list of :class:`ShadertoyChannel` objects. Each renderpass supports up to 4 inputs which then become .channel attributes.
            If used but not given, samples a black texture.

    Attributes:
        channels (list): A list of :class:`ShadertoyChannel` objects.
        _format (wgpu.TextureFormat): texture format for the render target.

    """
    def __init__(self, main:None, code: str, shader_type: str = "glsl", inputs: list = []):
        self._main = main
        self._shader_code = code
        self._shader_type = shader_type
        self._inputs = inputs
        self.channels = self._attach_inputs(inputs)

        # this is just a default - do we even need it?
        self._format: wgpu.TextureFormat = wgpu.TextureFormat.bgra8unorm

        self._prepare_render()

    @property
    def shader_code(self) -> str:
        """The shader code to use."""
        return self._shader_code

    @property
    def shader_type(self) -> str:
        """The shader type, automatically detected from the shader code, can be "wgsl" or "glsl"."""
        if self._shader_type in ("wgsl", "glsl"):
            return self._shader_type

        wgsl_main_expr = re.compile(r"fn(?:\s)+shader_main")
        glsl_main_expr = re.compile(r"void(?:\s)+(?:shader_main|mainImage)")
        if wgsl_main_expr.search(self.shader_code):
            return "wgsl"
        elif glsl_main_expr.search(self.shader_code):
            return "glsl"
        else:
            raise ValueError(
                "Could not find valid entry point function in shader code. Unable to determine if it's wgsl or glsl."
            )

    def _attach_inputs(self, inputs: list) -> List[ShadertoyChannel]:
        """
        Attach up to four input (channels) to a RenderPass.
        Handles cases where input is detected but not provided by falling back a 8x8 black texture.
        Also skips inputs that aren't used.
        Returns a list of `ShadertoyChannel` subclass instances to be set as .channels of the renderpass
        """

        if len(inputs) > 4:
            raise ValueError("Only 4 inputs supported")

        # fill up with None to always have 4 inputs.
        if len(inputs) < 4:
            inputs.extend([None] * (4 - len(inputs)))

        channel_pattern = re.compile(r"(?:iChannel|i_channel)(\d+)")
        detected_channels = [
            int(c) for c in set(channel_pattern.findall(self.common + self.shader_code))
        ]

        channels = []

        for inp_idx, inp in enumerate(inputs):
            if inp_idx not in detected_channels:
                channel = None
            elif type(inp) is ShadertoyChannel:
                # case where the base class is provided
                channel = inp.infer_subclass(parent=self, channel_idx=inp_idx)
            elif isinstance(inp, ShadertoyChannel):
                # case where a subclass is provided
                inp.channel_idx = inp_idx
                inp.parent = self
                channel = inp
            elif inp is None and inp_idx in detected_channels:
                # this is the base case where we sample the black texture.
                channel = ShadertoyChannelTexture(channel_idx=inp_idx)
            else:
                # do we even get here?
                channel = None

            # TODO: dynamic channels not yet implemented.
            # if channel is not None:
            #     self._input_headers += channel.get_header(shader_type=self.shader_type)
            channels.append(channel)

        return channels

    def _prepare_render(self):
        """
        Should be part of __init__ ?
        """
        pass


class ImageRenderPass(RenderPass):
    """
    The Image RenderPass of a Shadertoy. Renders to a canvas.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class BufferRenderPass(RenderPass):
    """
    The Buffer A-D RenderPass of a Shadertoy. Render to a texture that can be used as input for other renderpasses (include itself).
    Parameters:
        buffer_idx (str): one of "A", "B", "C" or "D". Required.
    """

    pass  # TODO at a later date


class CubemapRenderPass(RenderPass):
    """
    The Cube A RenderPass of a Shadertoy.
    this has slightly different headers see: https://shadertoyunofficial.wordpress.com/2016/07/20/special-shadertoy-features/
    """

    pass  # TODO at a later date


class SoundRenderPass(RenderPass):
    """
    The Sound RenderPass of a Shadertoy.
    sound is rendered to a buffer at the start and then played back. There is no interactivity....
    """

    pass  # TODO at a later date