import re
from typing import List

import numpy as np
import wgpu

from .inputs import ShadertoyChannel, ShadertoyChannelTexture

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
# TODO: avoid redundant globals, refactor to something like a headers.py file?
vertex_code_glsl_flipped = """#version 450 core

layout(location = 0) out vec2 vert_uv;

void main(void){
    int index = int(gl_VertexID);
    if (index == 0) {
        gl_Position = vec4(-1.0, -1.0, 0.0, 1.0);
        vert_uv = vec2(0.0, 0.0); // Flipped
    } else if (index == 1) {
        gl_Position = vec4(3.0, -1.0, 0.0, 1.0);
        vert_uv = vec2(2.0, 0.0); // Flipped
    } else {
        gl_Position = vec4(-1.0, 3.0, 0.0, 1.0);
        vert_uv = vec2(0.0, 2.0); // Flipped
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

// Shadertoy compatibility, see we can use the same code copied from shadertoy website

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

vertex_code_wgsl_flipped = """

struct Varyings {
    @builtin(position) position : vec4<f32>,
    @location(0) vert_uv : vec2<f32>,
};

@vertex
fn main(@builtin(vertex_index) index: u32) -> Varyings {
    var out: Varyings;
    if (index == u32(0)) {
        out.position = vec4<f32>(-1.0, -1.0, 0.0, 1.0);
        out.vert_uv = vec2<f32>(0.0, 0.0); // Flipped
    } else if (index == u32(1)) {
        out.position = vec4<f32>(3.0, -1.0, 0.0, 1.0);
        out.vert_uv = vec2<f32>(2.0, 0.0); // Flipped
    } else {
        out.position = vec4<f32>(-1.0, 3.0, 0.0, 1.0);
        out.vert_uv = vec2<f32>(0.0, -2.0); // Flipped
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
        main(Shadertoy): the main Shadertoy class of which this renderpass is part of.
        code (str): Shadercode for this buffer.
        shader_type(str): either "wgsl" or "glsl" can also be "auto" - which then gets solved by a regular expression, we should be able to match differnt renderpasses... Defaults to glsl
        inputs (list): A list of :class:`ShadertoyChannel` objects. Each pass supports up to 4 inputs/channels. If a channel is dected in the code but none provided, will be sampling a black texture.
    """

    # TODO: uniform data is per pass (as it includes iChannelResolution...)
    def __init__(
        self, code: str, main=None, shader_type: str = "glsl", inputs=[]
    ) -> None:
        self._main = main  # could be None...
        self._shader_type = shader_type
        self._shader_code = code
        self._inputs = inputs  # keep them here so we only attach them later?
        self._input_headers = ""
        # self.channels = self._attach_inputs(inputs)

    @property
    def main(self):  # -> "Shadertoy" (can't type due to circular import?)
        """
        The main Shadertoy class of which this renderpass is part of.
        """
        if self._main is None:
            raise ValueError("Main Shadertoy class is not set.")
        return self._main

    @main.setter
    def main(self, value):
        self._main = value

    @property
    def _uniform_data(self):
        """
        each RenderPass might have some differences in terms of times, and channel res...
        """
        return self.main._uniform_data

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
            self._shader_type = "wgsl"
        elif glsl_main_expr.search(self.shader_code):
            self._shader_type = "glsl"
        else:
            raise ValueError(
                "Could not find valid entry point function in shader code. Unable to determine if it's wgsl or glsl."
            )
        return self._shader_type

    def _update_textures(self, device: wgpu.GPUDevice) -> None:
        # self._uniform_data = self.main._uniform_data # force update?
        # print(f"{self._uniform_data['frame']} at start of _update_textures")
        device.queue.write_buffer(
            self._uniform_buffer,
            0,
            self._uniform_data.mem,
            0,
            self._uniform_data.nbytes,
        )

        for channel in self.channels:
            if channel is None or not channel.dynamic:
                continue

            layout, layout_entry = channel.bind_texture(device=device)

            self._binding_layout[channel.texture_binding] = layout[0]
            self._binding_layout[channel.sampler_binding] = layout[1]

            self._bind_groups_layout_entries[channel.texture_binding] = layout_entry[0]
            self._bind_groups_layout_entries[channel.sampler_binding] = layout_entry[1]

        self._finish_renderpass(device)

    def _attach_inputs(self, inputs: list) -> List[ShadertoyChannel]:
        if len(inputs) > 4:
            raise ValueError("Only 4 inputs supported")

        # fill up with None to always have 4 inputs.
        if len(inputs) < 4:
            inputs.extend([None] * (4 - len(inputs)))

        channel_pattern = re.compile(r"(?:iChannel|i_channel)(\d+)")
        detected_channels = [
            int(c)
            for c in set(channel_pattern.findall(self.main.common + self.shader_code))
        ]

        channels = []

        for inp_idx, inp in enumerate(inputs):
            if inp_idx not in detected_channels:
                channel = None
                # print(f"Channel {inp_idx} not used in shader code.")
                # maybe raise a warning or some error? For unusued channel
            elif type(inp) is ShadertoyChannel:
                channel = inp.infer_subclass(parent=self, channel_idx=inp_idx)
            elif isinstance(inp, ShadertoyChannel):
                inp.channel_idx = inp_idx
                inp.parent = self
                channel = inp
            elif inp is None and inp_idx in detected_channels:
                # this is the base case where we sample the black texture.
                channel = ShadertoyChannelTexture(channel_idx=inp_idx)
            else:
                # do we even get here?
                channel = None

            if channel is not None:
                self._input_headers += channel.get_header(shader_type=self.shader_type)
            channels.append(channel)

        return channels

    def prepare_render(self, device: wgpu.GPUDevice) -> None:
        # Step 1: compose shader programs
        self.channels = self._attach_inputs(self._inputs)
        shader_type = self.shader_type
        if shader_type == "glsl":
            if type(self) is BufferRenderPass:
                # TODO: figure out a one line manipulation (via comments and #define)?
                vertex_shader_code = vertex_code_glsl_flipped
            else:
                vertex_shader_code = vertex_code_glsl
            frag_shader_code = (
                builtin_variables_glsl
                + self._input_headers
                + self.main.common
                + self.shader_code
                + fragment_code_glsl
            )
        elif shader_type == "wgsl":
            if type(self) is BufferRenderPass:
                vertex_shader_code = vertex_code_wgsl_flipped
            else:
                vertex_shader_code = vertex_code_wgsl
            frag_shader_code = (
                builtin_variables_wgsl
                + self._input_headers
                + self.main.common
                + self.shader_code
                + fragment_code_wgsl
            )

        self._vertex_shader_program = device.create_shader_module(
            label="triangle_vert", code=vertex_shader_code
        )
        self._frag_shader_program = device.create_shader_module(
            label="triangle_frag", code=frag_shader_code
        )

        # Step 2: map uniform data to buffer
        self._uniform_buffer = device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Step 3: layout and bind groups
        self._bind_groups_layout_entries = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._uniform_buffer,
                    "offset": 0,
                    "size": self._uniform_data.nbytes,
                },
            },
        ]

        self._binding_layout = [
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            },
        ]

        # Step 4: add inputs as textures.
        channel_res = []
        for channel in self.channels:
            if channel is None:
                channel_res.extend([0, 0, 1, -99])  # default values; quick hack
                continue

            layout, layout_entry = channel.bind_texture(device=device)

            self._binding_layout.extend(layout)

            self._bind_groups_layout_entries.extend(layout_entry)
            channel_res.extend(channel.channel_res)  # padding/tests

        self._uniform_data["channel_res"] = tuple(channel_res)

        self._finish_renderpass(device)

    def _finish_renderpass(self, device: wgpu.GPUDevice) -> None:
        bind_group_layout = device.create_bind_group_layout(
            entries=self._binding_layout
        )

        self._bind_group = device.create_bind_group(
            layout=bind_group_layout,
            entries=self._bind_groups_layout_entries,
        )

        self._render_pipeline = device.create_render_pipeline(
            layout=device.create_pipeline_layout(
                bind_group_layouts=[bind_group_layout]
            ),
            vertex={
                "module": self._vertex_shader_program,
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
                "module": self._frag_shader_program,
                "entry_point": "main",
                "targets": [
                    {
                        "format": self._format,
                        "blend": None,  # maybe fine?
                    },
                ],
            },
        )


class ImageRenderPass(RenderPass):
    """
    The Image RenderPass of a Shadertoy.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._format = wgpu.TextureFormat.bgra8unorm
        # TODO figure out if there is anything specific. Maybe the canvas stuff? perhaps that should stay in the main class...

    def draw_image(self, device: wgpu.GPUDevice, present_context) -> None:
        """
        Draws the main image pass to the screen.
        """
        # maybe have an internal self._update for the uniform buffer too?
        self._update_textures(device)
        command_encoder = device.create_command_encoder()
        current_texture = present_context.get_current_texture()

        # TODO: maybe use a different name in this case?
        render_pass = command_encoder.begin_render_pass(
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
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(3, 1, 0, 0)
        render_pass.end()

        device.queue.submit([command_encoder.finish()])


class BufferRenderPass(RenderPass):
    """
    The Buffer A-D RenderPass of a Shadertoy.
    Parameters:
        buffer_idx (str): one of "A", "B", "C" or "D". Required.
    """

    def __init__(self, buffer_idx: str = "", **kwargs):
        super().__init__(**kwargs)
        self._buffer_idx = buffer_idx
        self._format = wgpu.TextureFormat.rgba32float

    @property
    def buffer_idx(self) -> str:
        if not self._buffer_idx:  # checks for empty string
            raise ValueError("Buffer index not set")
        return self._buffer_idx.lower()

    @buffer_idx.setter
    def buffer_idx(self, value: str):
        if value.lower() not in "abcd":
            raise ValueError("Buffer index must be one of 'A', 'B', 'C' or 'D'")
        self._buffer_idx = value

    @property
    def texture_size(self) -> tuple:
        """
        (columns, rows, 1), cols aligned to 64.
        (height)
        """
        if not hasattr(self, "_texture_size"):
            columns = self._pad_columns(int(self.main.resolution[0]))
            rows = int(self.main.resolution[1])
            self._texture_size = (columns, rows, 1)
        else:
            self._texture_size = self._texture.size

        return self._texture_size

    def _pad_columns(self, cols: int, alignment=16) -> int:
        if cols % alignment != 0:
            cols = (cols // alignment + 1) * alignment
        return cols

    def resize(self, new_cols: int, new_rows: int) -> None:
        """
        resizes the buffer to a new speicified size.
        Downscaling keeps the bottom left corner,
        upscaling pads the top and right with black.
        """
        # TODO: could this be redone as a compute shader?

        old_cols, old_rows, _ = self.texture_size
        new_cols = self._pad_columns(new_cols)
        if new_cols == old_cols and new_rows == old_rows:
            return
        old = self._download_texture()
        if new_rows < old_rows or new_cols < old_cols:
            new = old[:new_rows, :new_cols]
        else:
            new = np.pad(
                old, ((0, new_rows - old_rows), (0, new_cols - old_cols), (0, 0))
            )
        self._upload_texture(new)

    @property
    def texture(self) -> wgpu.GPUTexture:
        """
        the texture that the buffer renders to, will also be used as a texture by the BufferChannels.
        """
        if not hasattr(self, "_texture"):
            # creates the initial texture
            self._texture = self.main._device.create_texture(
                size=self.texture_size,
                format=self._format,
                usage=wgpu.TextureUsage.COPY_SRC
                | wgpu.TextureUsage.COPY_DST
                | wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.TEXTURE_BINDING,
            )
        return self._texture

    def draw_buffer(self, device: wgpu.GPUDevice) -> None:
        """
        draws the buffer to the texture and updates self._texture.
        """
        self._update_textures(device)
        command_encoder = device.create_command_encoder()

        # create a temporary texture as a render target
        target_texture = device.create_texture(
            size=self.texture_size,
            format=self._format,
            usage=wgpu.TextureUsage.COPY_SRC | wgpu.TextureUsage.RENDER_ATTACHMENT,
        )

        # TODO: maybe use a different name in this case?
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_texture.create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )

        render_pass.set_pipeline(self._render_pipeline)
        render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        render_pass.draw(3, 1, 0, 0)  # what is .draw_indirect?
        render_pass.end()

        command_encoder.copy_texture_to_texture(
            {
                "texture": target_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "texture": self._texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            self.texture_size,  # could this handle resizing?
        )

        device.queue.submit([command_encoder.finish()])

    def _download_texture(
        self,
        device: wgpu.GPUDevice = None,
        size=None,
        command_encoder: wgpu.GPUCommandEncoder = None,
    ):
        """
        downloads the texture from the GPU to the CPU by returning a numpy array.
        """
        if device is None:
            device = self.main._device
        if command_encoder is None:
            command_encoder = device.create_command_encoder()
        if size is None:
            size = self.texture_size

        buffer = device.create_buffer(
            size=(size[0] * size[1] * 16),
            usage=wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST,
        )
        command_encoder.copy_texture_to_buffer(
            {
                "texture": self.texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": buffer,
                "offset": 0,
                "bytes_per_row": size[0] * 16,
                "rows_per_image": size[1],
            },
            size,
        )
        device.queue.submit([command_encoder.finish()])
        frame = device.queue.read_buffer(buffer)
        frame = np.frombuffer(frame, dtype=np.float32).reshape(size[1], size[0], 4)
        # redundant copy?
        # self._last_frame = frame
        return frame

    def _upload_texture(self, data, device=None, command_encoder=None):
        """
        uploads some data to self._texture.
        """
        if device is None:
            device = self.main._device
        if command_encoder is None:
            command_encoder = device.create_command_encoder()

        data = np.ascontiguousarray(data)
        # create a new texture with the changed size?
        new_texture = device.create_texture(
            size=(data.shape[1], data.shape[0], 1),
            format=self._format,
            usage=wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.RENDER_ATTACHMENT
            | wgpu.TextureUsage.COPY_DST
            | wgpu.TextureUsage.TEXTURE_BINDING,
        )

        device.queue.write_texture(
            {
                "texture": new_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            data,
            {
                "offset": 0,
                "bytes_per_row": data.shape[1] * 16,
                "rows_per_image": data.shape[0],
            },
            (data.shape[1], data.shape[0], 1),
        )
        self._texture = new_texture


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
