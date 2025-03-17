import re
from typing import List

import wgpu

from .inputs import ShadertoyChannel, ShadertoyChannelTexture

builtin_variables_glsl = """#version 450 core

vec4 iMouse;
vec4 iDate;
vec3 iResolution;
float iTime;
vec3 iChannelResolution[4];
float iTimeDelta;
int iFrame;
float iFrameRate;
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


class RenderPass:
    """
    Base class for renderpass in a Shadertoy.
    Parameters:
        code (str): Shadercode for this renderpass.
        main (Shadertoy): the main `Shadertoy` class of which this renderpass is part of. Defaults to None.
        shader_type (str): either "wgsl" or "glsl" can also be "auto" - which then gets solved by a regular expression.
            Defaults to "glsl".
        inputs (list): A list of :class:`ShadertoyChannel` objects. Each renderpass supports up to 4 inputs which then become .channel attributes.
            If used but not given, samples a black texture.
    """

    def __init__(
        self, code: str, main=None, shader_type: str = "glsl", inputs: list = []
    ):
        self._shader_code = code
        self._main = main
        self._shader_type = shader_type
        # we keep track of the inputs before we can attach them as channels.
        self._inputs = inputs
        self._input_headers = ""

        # this is just a default - do we even need it?
        self._format: wgpu.TextureFormat = wgpu.TextureFormat.bgra8unorm

        # the render can only be prepared when main is set
        if main is not None:
            self._prepare_render()
        # as long as main is not set, this renderpass is not ready to be used.

    def get_current_texture(self) -> wgpu.GPUTexture:
        """
        The current (next) texture to draw to
        """
        raise NotImplementedError("This method should be implemented by subclasses")

    @property
    def shader_code(self) -> str:
        """The shader code to use."""
        return self._shader_code

    @property
    def main(self):  # -> "Shadertoy": #TODO: how can be get this type hint?
        if self._main is not None:
            return self._main
        else:
            raise AttributeError(
                "main not set yet and the renderpass isn't ready to be used"
            )

    @main.setter
    def main(self, main_cls):
        """
        Register the main shadertoy class for this renderpass.
        Also trigger _prepare_render() to finish initialization. (moving this to first draw)
        """
        self._main = main_cls
        # self._prepare_render()

    @property
    def _device(self) -> wgpu.GPUDevice:
        return self.main._device

    @property
    def shader_type(self) -> str:
        """The shader type, automatically detected from the shader code, can be "wgsl" or "glsl"."""
        if self._shader_type not in ("wgsl", "glsl"):
            wgsl_main_expr = re.compile(r"fn(?:\s)+shader_main")
            glsl_main_expr = re.compile(r"void(?:\s)+mainImage")
            if wgsl_main_expr.search(self.shader_code):
                self._shader_type = "wgsl"
            elif glsl_main_expr.search(self.shader_code):
                self._shader_type = "glsl"
            else:
                raise ValueError(
                    "Could not find valid entry point function in shader code. Unable to determine if it's wgsl or glsl."
                )
        return self._shader_type

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
            int(c)
            for c in set(channel_pattern.findall(self.main.common + self.shader_code))
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

            if channel is not None:
                self._input_headers += channel.make_header(shader_type=self.shader_type)
            channels.append(channel)

        return channels

    def _prepare_render(self):
        """
        This private method can only be called after the main Shadertoy class is set.
        It attaches inputs, assembles the shadercode and creates the render pipeline.
        """

        # inputs can only be attached once the main class is set, so calling it here should do it.
        self.channels = self._attach_inputs(self._inputs)
        vertex_shader_code, frag_shader_code = self.construct_code()

        vertex_shader_program = self._device.create_shader_module(
            label="shadertoy_vertex", code=vertex_shader_code
        )
        frag_shader_program = self._device.create_shader_module(
            label="shadertoy_fragment", code=frag_shader_code
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

    def draw(self) -> wgpu.GPUCommandBuffer:
        """
        Updates uniforms and encodes the draw calls for this renderpass.
        Returns the command buffer.
        """

        if not hasattr(self, "_render_pipeline"):
            # basically this needs to be done before the first draw. (but we don't need to check this every single frame -.-)
            self._prepare_render()

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

        command_encoder: wgpu.GPUCommandEncoder = self._device.create_command_encoder()
        current_texture: wgpu.GPUTexture = self.get_current_texture()

        render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
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

    def construct_code(self) -> tuple[str, str]:
        """
        Public method to get the full vertex and fragment code for this renderpass.
        assembles the code templates for the vertext and fragment stages.
        """
        # left public since it has use outside of using it in the renderpass,
        # for example getting valid glsl from just a shadertoy image pass to use in naga validation.

        if self.shader_type == "glsl":
            vertex_shader_code = """
                #version 450 core
                vec2 pos[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
                void main() {
                    int index = int(gl_VertexID);
                    gl_Position = vec4(pos[index], 0.0, 1.0);
                }
                """
            # the image pass needs to be yflipped, buffers not. However dFdy is still violated. (https://github.com/pygfx/shadertoy/issues/38)
            fragment_code_glsl = f"""
                uniform struct ShadertoyInput {{
                    vec4 si_mouse;
                    vec4 si_date;
                    vec3 si_resolution;
                    float si_time;
                    vec3 si_channel_res[4];
                    float si_time_delta;
                    int si_frame;
                    float si_framerate;
                }};

                layout(binding = 0) uniform ShadertoyInput input;
                out vec4 FragColor;
                void main(){{
                    iMouse = input.si_mouse;
                    iDate = input.si_date;
                    iResolution = input.si_resolution;
                    iTime = input.si_time;
                    iChannelResolution = input.si_channel_res;
                    iTimeDelta = input.si_time_delta;
                    iFrame = input.si_frame;
                    iFrameRate = input.si_framerate;
                    
                    // handle the YFLIP part for just the Image pass?
                    vec2 fragcoord=vec2(gl_FragCoord.x, {"iResolution.y-" if isinstance(self, ImageRenderPass) else ""}gl_FragCoord.y);
                    mainImage(FragColor, fragcoord);
                }}
                """
            frag_shader_code = (
                builtin_variables_glsl
                + self._input_headers
                + self.main.common
                + self.shader_code
                + fragment_code_glsl
            )
        elif self.shader_type == "wgsl":
            vertex_shader_code = """
                struct VertexOut {
                    @builtin(position) position : vec4<f32>
                }
                
                @vertex
                fn main(@builtin(vertex_index) vertIndex: u32) -> VertexOut {
                    var pos = array(
                        vec2<f32>(-1.0, -1.0),
                        vec2<f32>(3.0, -1.0),
                        vec2<f32>(-1.0, 3.0)
                    );
                    var out: VertexOut;
                    out.position = vec4<f32>(pos[vertIndex], 0.0, 1.0);
                    return out;
                }
                """
            fragment_code_wgsl = f"""
                struct ShadertoyInput {{
                    si_mouse: vec4<f32>,
                    si_date: vec4<f32>,
                    si_resolution: vec3<f32>,
                    si_time: f32,
                    si_channel_res: array<vec4<f32>,4>,
                    si_time_delta: f32,
                    si_frame: u32,
                    si_framerate: f32,
                }};
                struct VertexOut {{
                    @builtin(position) position : vec4<f32>,
                }};
                @group(0) @binding(0)
                var<uniform> input: ShadertoyInput;
                @fragment
                fn main(in: VertexOut) -> @location(0) vec4<f32> {{
                    i_mouse = input.si_mouse;
                    i_date = input.si_date;
                    i_resolution = input.si_resolution;
                    i_time = input.si_time;
                    i_channel_resolution = input.si_channel_res;
                    i_time_delta = input.si_time_delta;
                    i_frame = input.si_frame;
                    i_framerate = input.si_framerate;

                    // Yflip for the image pass (not the correct solution)
                    let frag_coord = vec2f(in.position.x, {"i_resolution.y-" if isinstance(self, ImageRenderPass) else ""}in.position.y);
                    return shader_main(frag_coord);
                }}
                """
            frag_shader_code = (
                builtin_variables_wgsl
                + self._input_headers
                + self.main.common
                + self.shader_code
                + fragment_code_wgsl
            )
        return vertex_shader_code, frag_shader_code


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

    def __init__(self, buffer_idx: str, **kwargs):
        super().__init__(**kwargs)
        
        self.buffer_idx = buffer_idx.lower()
        if self.buffer_idx not in "abcd":
            raise ValueError("buffer_idx must be one of 'A', 'B', 'C' or 'D'")
        
        self._texture_front = None
        self._texture_back = None
        self.format = wgpu.TextureFormat.rgba32float # requieres the feature wgpu.FeatureName.float32_filterable

    def get_current_texture(self) -> wgpu.GPUTexture:
        """
        The current (next) texture to draw to
        """
        # for the buffer pass we always draw the `back` and read from the `front` ?
        return self._texture_back

    def draw(self) -> wgpu.GPUCommandBuffer:
        """
        the draw function for the buffer needs to additionally swap the textures
        """
        super().draw()
        # swap the textures
        self._texture_front, self._texture_back = self._texture_back, self._texture_front
        # update all bind groups? (is done in the draw_image function of the main class)

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
