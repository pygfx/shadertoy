import re
from typing import List

import numpy as np
import wgpu

from .inputs import ShadertoyChannel, ShadertoyChannelTexture

# TODO: simplify all the shader code snippets
vertex_code_glsl = """#version 450 core

layout(location = 0) out vec2 vert_uv;

void main(void){
    int index = int(gl_VertexID);
    if (index == 0) {
        gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
        vert_uv = vec2(0.0, 1.0);
    } else if (index == 1) {
        gl_Position = vec4(3.0, -1.0, 0.0, 1.0);
        vert_uv = vec2(2.0, 1.0);
    } else {
        gl_Position = vec4(-1.0, 3.0, 0.0, 1.0);
        vert_uv = vec2(0.0, -1.0);
    }
}
"""


builtin_variables_glsl = """#version 450 core

vec4 i_mouse;
vec4 i_date;
vec3 i_resolution;
float i_time;
vec3 i_channel_resolution[4];
float i_time_delta;
int i_frame;
float i_framerate;

layout(binding = 1) uniform texture2D i_channel0;
layout(binding = 2) uniform sampler sampler0;
layout(binding = 3) uniform texture2D i_channel1;
layout(binding = 4) uniform sampler sampler1;
layout(binding = 5) uniform texture2D i_channel2;
layout(binding = 6) uniform sampler sampler2;
layout(binding = 7) uniform texture2D i_channel3;
layout(binding = 8) uniform sampler sampler3;

// Shadertoy compatibility, see we can use the same code copied from shadertoy website

#define iChannel0 sampler2D(i_channel0, sampler0)
#define iChannel1 sampler2D(i_channel1, sampler1)
#define iChannel2 sampler2D(i_channel2, sampler2)
#define iChannel3 sampler2D(i_channel3, sampler3)

#define iMouse i_mouse
#define iDate i_date
#define iResolution i_resolution
#define iTime i_time
#define iChannelResolution i_channel_resolution
#define iTimeDelta i_time_delta
#define iFrame i_frame
#define iFrameRate i_framerate

#define mainImage shader_main
"""


fragment_code_glsl = """
layout(location = 0) in vec2 vert_uv;

struct ShadertoyInput {
    vec4 si_mouse;
    vec4 si_date;
    vec3 si_resolution;
    float si_time;
    vec3 si_channel_res[4];
    float si_time_delta;
    int si_frame;
    float si_framerate;
};

layout(binding = 0) uniform ShadertoyInput input;
out vec4 FragColor;
void main(){

    i_mouse = input.si_mouse;
    i_date = input.si_date;
    i_resolution = input.si_resolution;
    i_time = input.si_time;
    i_channel_resolution = input.si_channel_res;
    i_time_delta = input.si_time_delta;
    i_frame = input.si_frame;
    i_framerate = input.si_framerate;
    vec2 frag_uv = vec2(vert_uv.x, 1.0 - vert_uv.y);
    vec2 frag_coord = frag_uv * i_resolution.xy;

    shader_main(FragColor, frag_coord);

}

"""


vertex_code_wgsl = """

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) vert_uv : vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) index: u32) -> Varyings {
    var out: Varyings;
    if (index == u32(0)) {
        out.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.vert_uv = vec2<f32>(0.0, 1.0);
    } else if (index == u32(1)) {
        out.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
        out.vert_uv = vec2<f32>(2.0, 1.0);
    } else {
        out.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        out.vert_uv = vec2<f32>(0.0, -1.0);
    }
    return out;

}
"""


builtin_variables_wgsl = """

var<private> i_mouse: vec4<f32>;
var<private> i_date: vec4<f32>;
var<private> i_resolution: vec3<f32>;
var<private> i_time: f32;
var<private> i_channel_resolution: array<vec4<f32>,4>;
var<private> i_time_delta: f32;
var<private> i_frame: u32;
var<private> i_framerate: f32;

// TODO: more global variables
// var<private> i_frag_coord: vec2<f32>;

"""


fragment_code_wgsl = """

struct ShadertoyInput {
    si_mouse: vec4<f32>,
    si_date: vec4<f32>,
    si_resolution: vec3<f32>,
    si_time: f32,
    si_channel_res: array<vec4<f32>,4>,
    si_time_delta: f32,
    si_frame: u32,
    si_framerate: f32,
};

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) vert_uv : vec2<f32>,
};

@group(0) @binding(0)
var<uniform> input: ShadertoyInput;

@group(0) @binding(1)
var i_channel0: texture_2d<f32>;
@group(0) @binding(3)
var i_channel1: texture_2d<f32>;
@group(0) @binding(5)
var i_channel2: texture_2d<f32>;
@group(0) @binding(7)
var i_channel3: texture_2d<f32>;

@group(0) @binding(2)
var sampler0: sampler;
@group(0) @binding(4)
var sampler1: sampler;
@group(0) @binding(6)
var sampler2: sampler;
@group(0) @binding(8)
var sampler3: sampler;

@fragment
fn main(in: Varyings) -> @location(0) vec4<f32> {

    i_mouse = input.si_mouse;
    i_date = input.si_date;
    i_resolution = input.si_resolution;
    i_time = input.si_time;
    i_channel_resolution = input.si_channel_res;
    i_time_delta = input.si_time_delta;
    i_frame = input.si_frame;
    i_framerate = input.si_framerate;
    let frag_uv = vec2<f32>(in.vert_uv.x, 1.0 - in.vert_uv.y);
    let frag_coord = frag_uv * i_resolution.xy;

    return shader_main(frag_coord);
}

"""


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
    def main(self) -> "Shadertoy":
        if self._main is not None:
            return self._main
        else:
            raise AttributeError("_main not set yet")

    @property
    def _device(self) -> wgpu.GPUDevice:
        return self.main._device

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
            int(c) for c in set(channel_pattern.findall(self.main.common + self.shader_code))
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

            # TODO: dynamic channel headers not yet implemented.
            # if channel is not None:
            #     self._input_headers += channel.get_header(shader_type=self.shader_type)
            channels.append(channel)

        return channels

    def _prepare_render(self):
        # First assemble the shader code
        shader_type = self.shader_type
        if shader_type == "glsl":
            vertex_shader_code = vertex_code_glsl
            frag_shader_code = (
                builtin_variables_glsl
                + self.main.common
                + self.shader_code
                + fragment_code_glsl
            )
        elif shader_type == "wgsl":
            vertex_shader_code = vertex_code_wgsl
            frag_shader_code = (
                builtin_variables_wgsl
                + self.main.common
                + self.shader_code
                + fragment_code_wgsl
            )

        vertex_shader_program = self._device.create_shader_module(
            label="triangle_vert", code=vertex_shader_code
        )
        frag_shader_program = self._device.create_shader_module(
            label="triangle_frag", code=frag_shader_code
        )

        # Uniforms are mainly global so the ._uniform_data object can be copied into a local uniform buffer
        self._uniform_buffer = self._device.create_buffer(
            size=self.main._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        bind_groups_layout_entries = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._uniform_buffer,
                    "offset": 0,
                    "size": self.main._uniform_data.nbytes,
                },
            },
        ]

        binding_layout = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]

        # setup bind groups for the channels
        channel_res = []
        for channel in self.channels:
            if channel is None:
                channel_res.extend([0, 0, 1, -99])  # default values; quick hack
                continue
            layout, layout_entry = channel.bind_texture(device=self._device)
            binding_layout.extend(layout)
            bind_groups_layout_entries.extend(layout_entry)
            channel_res.extend(channel.channel_res)

        # this uniform data should be per renderpass
        self._channel_res = tuple(channel_res)
        bind_group_layout = self._device.create_bind_group_layout(
            entries=binding_layout
        )

        self._bind_group = self._device.create_bind_group(
            layout=bind_group_layout,
            entries=bind_groups_layout_entries,
        )

        self._render_pipeline = self._device.create_render_pipeline(
            layout=self._device.create_pipeline_layout(
                bind_group_layouts=[bind_group_layout]
            ),
            vertex={
                "module": vertex_shader_program,
                "entry_point": "main",
                "buffers": [],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "front_face": wgpu.FrontFace.ccw,
                "cull_mode": wgpu.CullMode.none,
            },
            depth_stencil=None,
            multisample=None,
            fragment={
                "module": frag_shader_program,
                "entry_point": "main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.bgra8unorm,
                    },
                ],
            },
        )

    # can this be generalized?
    def draw(self) -> wgpu.GPUCommandBuffer:
        """
        Updates uniforms and encodes the draw calls for this renderpass.
        Returns the command buffer.
        """
        # to keep channel_res per renderpass - we need to overwrite it? (really lazy implementation)
        # channel_res can change with resizing, so it's not neccassarily constant
        # we might be able to reorder the layout and then cleverly use offsets
        # TODO: consider push constants https://github.com/pygfx/wgpu-py/pull/574
        self.main._uniform_data["channel_res"] = self._channel_res
        self._device.queue.write_buffer(
            buffer=self._uniform_buffer,
            buffer_offset=0,
            data=self.main._uniform_data.mem,
            data_offset=0,
            size=self.main._uniform_data.nbytes,
        )

        command_encoder:wgpu.GPUCommandEncoder = self._device.create_command_encoder()
        current_texture:wgpu.GPUTexture = self.get_current_texture()

        render_pass:wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": current_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(self._render_pipeline)
        # self._bind_group might get generalized out for buffer
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()

        return command_encoder.finish()

class ImageRenderPass(RenderPass):
    """
    The Image RenderPass of a Shadertoy. Renders to a canvas.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_current_texture(self) -> wgpu.GPUTexture:
        """
        The current (next) texture to draw to
        """
        # for the image pass this swap chain is handled by the context/canvas
        return self.main._present_context.get_current_texture()

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