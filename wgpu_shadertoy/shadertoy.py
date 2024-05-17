import collections
import ctypes
import os
import re
import time

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas
from wgpu.gui.offscreen import run as run_offscreen

from .api import shader_args_from_json, shadertoy_from_id
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


class UniformArray:
    """Convenience class to create a uniform array.

    Maybe we can make it a public util at some point.
    Ensure that the order matches structs in the shader code.
    See https://www.w3.org/TR/WGSL/#alignment-and-size for reference on alignment.
    """

    def __init__(self, *args):
        # Analyse incoming fields
        fields = []
        byte_offset = 0
        for name, format, n in args:
            assert format in ("f", "i", "I")
            field = name, format, byte_offset, byte_offset + n * 4
            fields.append(field)
            byte_offset += n * 4
        # Get padding
        nbytes = byte_offset
        while nbytes % 16:
            nbytes += 1
        # Construct memoryview object and a view for each field
        self._mem = memoryview((ctypes.c_uint8 * nbytes)()).cast("B")
        self._views = {}
        for name, format, i1, i2 in fields:
            self._views[name] = self._mem[i1:i2].cast(format)

    @property
    def mem(self):
        return self._mem

    @property
    def nbytes(self):
        return self._mem.nbytes

    def __getitem__(self, key):
        v = self._views[key].tolist()
        return v[0] if len(v) == 1 else v

    def __setitem__(self, key, val):
        m = self._views[key]
        n = m.shape[0]
        if n == 1:
            assert isinstance(val, (float, int))
            m[0] = val
        else:
            assert isinstance(val, (tuple, list))
            for i in range(n):
                m[i] = val[i]


class Shadertoy:
    """Provides a "screen pixel shader programming interface" similar to `shadertoy <https://www.shadertoy.com/>`_.

    It helps you research and quickly build or test shaders using `WGSL` or `GLSL` via WGPU.

    Parameters:
        shader_code (str): The shader code to use.
        common (str): The common shaderpass code gets executed before all other shaderpasses (buffers/image/sound). Defaults to empty string.
        buffers (dict(str)): Codes for buffers A through D. Still requires to set buffer as channel input. Defaults to empty strings.
        resolution (tuple): The resolution of the shadertoy in (width, height). Defaults to (800, 450).
        shader_type (str): Can be "wgsl" or "glsl". On any other value, it will be automatically detected from shader_code. Default is "auto".
        offscreen (bool): Whether to render offscreen. Default is False.
        inputs (list): A list of :class:`ShadertoyChannel` objects. Supports up to 4 inputs. Defaults to sampling a black texture.
        title (str): The title of the window. Defaults to "Shadertoy".
        complete (bool): Whether the shader is complete. Unsupported renderpasses or inputs will set this to False. Default is True.

    The shader code must contain a entry point function:

    WGSL: ``fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{}``
    GLSL: ``void shader_main(out vec4 frag_color, in vec2 frag_coord){}``

    It has a parameter ``frag_coord`` which is the current pixel coordinate (in range 0..resolution, origin is bottom-left),
    and it must return a vec4<f32> color (for GLSL, it's the ``out vec4 frag_color`` parameter), which is the color of the pixel at that coordinate.

    some built-in variables are available in the shader:

    * ``i_mouse``: the mouse position in pixels
    * ``i_date``: the current date and time as a vec4 (year, month, day, seconds)
    * ``i_resolution``: the resolution of the shadertoy
    * ``i_time``: the global time in seconds
    * ``i_time_delta``: the time since last frame in seconds
    * ``i_frame``: the frame number
    * ``i_framerate``: the number of frames rendered in the last second.

    For GLSL, you can also use the aliases ``iTime``, ``iTimeDelta``, ``iFrame``, ``iResolution``, ``iMouse``, ``iDate`` and ``iFrameRate`` of these built-in variables,
    the entry point function also has an alias ``mainImage``, so you can use the shader code copied from shadertoy website without making any changes.
    """

    # todo: add remaining built-in variables (i_channel_time)
    # todo: support multiple render passes (`i_channel0`, `i_channel1`, etc.)

    def __init__(
        self,
        shader_code: str,
        common: str = "",
        buffers: dict = {
            "a": "",
            "b": "",
            "c": "",
            "d": "",
        },  # maybe Default dict instead?
        resolution=(800, 450),
        shader_type="auto",
        offscreen=None,
        inputs=[None] * 4,
        title: str = "Shadertoy",
        complete: bool = True,
    ) -> None:
        self._uniform_data = UniformArray(
            ("mouse", "f", 4),
            ("date", "f", 4),
            ("resolution", "f", 3),
            ("time", "f", 1),
            ("channel_res", "f", (3 + 1) * 4),  # vec3 + 1 padding, 4 channels
            ("time_delta", "f", 1),
            ("frame", "I", 1),
            ("framerate", "f", 1),
        )

        self._shader_code = shader_code
        self.common = common + "\n"

        self.buffers = {"a": "", "b": "", "c": "", "d": ""}
        for k, v in buffers.items():
            k = k.lower()[-1]
            if k not in "abcd":
                raise ValueError(f"Invalid buffer key: {k}")
            self.buffers[k] = v

        self._uniform_data["resolution"] = (*resolution, 1)
        self._shader_type = shader_type.lower()

        # if no explicit offscreen option was given
        # inherit wgpu-py force offscreen option
        if offscreen is None and os.environ.get("WGPU_FORCE_OFFSCREEN") == "true":
            offscreen = True
        self._offscreen = offscreen

        if len(inputs) < 4:
            inputs.extend([None] * (4 - len(inputs)))
            # likely a better solution. But theoretically, someone could set one or more inputs but still mention a channel beyond that.

        self.channels = [None] * 4
        channel_pattern = re.compile(
            r"(?:iChannel|i_channel)(\d+)"
        )  # non capturing group is important!

        # TODO: redo this whole logic as channel_idx is available from the api
        # so we only need to assign it if it's not set, like when using the classes directly.
        for channel_idx in channel_pattern.findall(shader_code):
            channel_idx = int(channel_idx)
            if channel_idx not in (0, 1, 2, 3):
                raise ValueError(
                    f"Only iChannel0 to iChannel3 are supported. Found {channel_idx=}"
                )
            if inputs[channel_idx] is None:
                self.channels[channel_idx] = ShadertoyChannelTexture(
                    channel_idx=channel_idx
                )
            elif type(inputs[channel_idx]) is ShadertoyChannel:
                self.channels[channel_idx] = inputs[channel_idx].infer_subclass(
                    main=self
                )
            elif isinstance(inputs[channel_idx], ShadertoyChannel):
                self.channels[channel_idx] = inputs[channel_idx]
            else:
                raise ValueError(
                    f"Invalid input type for channel {channel_idx=} - {inputs[channel_idx]=}"
                )
            self.channels[channel_idx].channel_idx = channel_idx  # redundant?

        self.title = title
        self.complete = complete

        if not self.complete:
            self.title += " (incomplete)"

        self._prepare_render()
        self._bind_events()

    @property
    def resolution(self):
        """The resolution of the shadertoy as a tuple (width, height) in pixels."""
        return tuple(self._uniform_data["resolution"])[:2]

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

    @classmethod
    def from_json(cls, dict_or_path, **kwargs):
        """Builds a `Shadertoy` instance from a JSON-like dict of Shadertoy.com shader data."""
        shader_args = shader_args_from_json(dict_or_path, **kwargs)
        return cls(**shader_args)

    @classmethod
    def from_id(cls, id_or_url: str, **kwargs):
        """Builds a `Shadertoy` instance from a Shadertoy.com shader id or url. Requires API key to be set."""
        shader_data = shadertoy_from_id(id_or_url)
        return cls.from_json(shader_data, **kwargs)

    def _prepare_render(self):
        import wgpu.backends.auto

        if self._offscreen:
            self._canvas = OffscreenCanvas(
                title=self.title, size=self.resolution, max_fps=60
            )
        else:
            self._canvas = WgpuCanvas(
                title=self.title, size=self.resolution, max_fps=60
            )

        self._device = wgpu.utils.device.get_default_device()

        self._present_context = self._canvas.get_context()

        # We use "bgra8unorm" not "bgra8unorm-srgb" here because we want to let the shader fully control the color-space.
        self._present_context.configure(
            device=self._device, format=wgpu.TextureFormat.bgra8unorm
        )

        shader_type = self.shader_type
        if shader_type == "glsl":
            vertex_shader_code = vertex_code_glsl
            frag_shader_code = (
                builtin_variables_glsl
                + self.common
                + self.shader_code
                + fragment_code_glsl
            )
        elif shader_type == "wgsl":
            vertex_shader_code = vertex_code_wgsl
            frag_shader_code = (
                builtin_variables_wgsl
                + self.common
                + self.shader_code
                + fragment_code_wgsl
            )

        vertex_shader_program = self._device.create_shader_module(
            label="triangle_vert", code=vertex_shader_code
        )
        frag_shader_program = self._device.create_shader_module(
            label="triangle_frag", code=frag_shader_code
        )

        self._uniform_buffer = self._device.create_buffer(
            size=self._uniform_data.nbytes,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        bind_groups_layout_entries = [
            {
                "binding": 0,
                "resource": {
                    "buffer": self._uniform_buffer,
                    "offset": 0,
                    "size": self._uniform_data.nbytes,
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
        channel_res = []
        for channel_idx, channel in enumerate(self.channels):
            if channel is None:
                channel_res.extend([0, 0, 1, -99])  # default values; quick hack
                continue
            texture_binding = (2 * channel_idx) + 1
            sampler_binding = 2 * (channel_idx + 1)
            binding_layout.extend(
                [
                    {
                        "binding": texture_binding,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {
                            "sample_type": wgpu.TextureSampleType.float,
                            "view_dimension": wgpu.TextureViewDimension.d2,
                        },
                    },
                    {
                        "binding": sampler_binding,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                ]
            )

            texture = self._device.create_texture(
                size=channel.texture_size,
                format=wgpu.TextureFormat.rgba8unorm,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            texture_view = texture.create_view()

            self._device.queue.write_texture(
                {
                    "texture": texture,
                    "origin": (0, 0, 0),
                    "mip_level": 0,
                },
                channel.data,
                {
                    "offset": 0,
                    "bytes_per_row": channel.bytes_per_pixel
                    * channel.size[1],  # must be multiple of 256?
                    "rows_per_image": channel.size[0],  # same is done internally
                },
                texture.size,
            )

            sampler = self._device.create_sampler(**channel.sampler_settings)
            bind_groups_layout_entries.extend(
                [
                    {
                        "binding": texture_binding,
                        "resource": texture_view,
                    },
                    {
                        "binding": sampler_binding,
                        "resource": sampler,
                    },
                ]
            )
            channel_res.append(channel.size[1])  # width
            channel_res.append(channel.size[0])  # height
            channel_res.append(1)  # always 1 for pixel aspect ratio
            channel_res.append(-99)  # padding/tests
        self._uniform_data["channel_res"] = tuple(channel_res)
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
                        "blend": {
                            "color": (
                                wgpu.BlendFactor.one,
                                wgpu.BlendFactor.zero,
                                wgpu.BlendOperation.add,
                            ),
                            "alpha": (
                                wgpu.BlendFactor.one,
                                wgpu.BlendFactor.zero,
                                wgpu.BlendOperation.add,
                            ),
                        },
                    },
                ],
            },
        )

    def _bind_events(self):
        def on_resize(event):
            w, h = event["width"], event["height"]
            self._uniform_data["resolution"] = (w, h, 1)

        def on_mouse_move(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                _, _, x2, y2 = self._uniform_data["mouse"]
                x1, y1 = event["x"], self.resolution[1] - event["y"]
                self._uniform_data["mouse"] = x1, y1, abs(x2), -abs(y2)

        def on_mouse_down(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                x, y = event["x"], self.resolution[1] - event["y"]
                self._uniform_data["mouse"] = (x, y, abs(x), abs(y))

        def on_mouse_up(event):
            if event["button"] == 1 or 1 in event["buttons"]:
                x1, y1, x2, y2 = self._uniform_data["mouse"]
                self._uniform_data["mouse"] = x1, y1, -abs(x2), -abs(y2)

        self._canvas.add_event_handler(on_resize, "resize")
        self._canvas.add_event_handler(on_mouse_move, "pointer_move")
        self._canvas.add_event_handler(on_mouse_down, "pointer_down")
        self._canvas.add_event_handler(on_mouse_up, "pointer_up")

    def _update(self):
        now = time.perf_counter()
        if not hasattr(self, "_last_time"):
            self._last_time = now

        if not hasattr(self, "_time_history"):
            self._time_history = collections.deque(maxlen=256)

        time_delta = now - self._last_time
        self._uniform_data["time_delta"] = time_delta
        self._last_time = now
        self._uniform_data["time"] += time_delta
        self._time_history.append(self._uniform_data["time"])

        self._uniform_data["framerate"] = sum(
            [1 for t in self._time_history if t > self._uniform_data["time"] - 1]
        )

        if not hasattr(self, "_frame"):
            self._frame = 0

        time_struct = time.localtime()
        self._uniform_data["date"] = (
            float(time_struct.tm_year),
            float(time_struct.tm_mon - 1),
            float(time_struct.tm_mday),
            time_struct.tm_hour * 3600
            + time_struct.tm_min * 60
            + time_struct.tm_sec
            + now % 1,
        )

        self._uniform_data["frame"] = self._frame
        self._frame += 1

    def _draw_frame(self):
        # Update uniform buffer
        self._update()
        self._device.queue.write_buffer(
            self._uniform_buffer,
            0,
            self._uniform_data.mem,
            0,
            self._uniform_data.nbytes,
        )

        command_encoder = self._device.create_command_encoder()
        current_texture = self._present_context.get_current_texture()

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

        self._device.queue.submit([command_encoder.finish()])

        self._canvas.request_draw()

    def show(self):
        self._canvas.request_draw(self._draw_frame)
        if self._offscreen:
            run_offscreen()
        else:
            run()

    def snapshot(self, time_float: float = 0.0, mouse_pos: tuple = (0, 0, 0, 0)):
        """
        Returns an image of the specified time. (Only available when ``offscreen=True``)

        Parameters:
            time_float (float): The time to snapshot. It essentially sets ``i_time`` to a specific number. (Default is 0.0)
            mouse_pos (tuple): The mouse position in pixels in the snapshot. It essentially sets ``i_mouse`` to a 4-tuple. (Default is (0,0,0,0))
        Returns:
            frame (memoryview): snapshot with transparency. This object can be converted to a numpy array (without copying data)
        using ``np.asarray(arr)``
        """
        if not self._offscreen:
            raise NotImplementedError("Snapshot is only available in offscreen mode.")

        if hasattr(self, "_last_time"):
            self.__delattr__("_last_time")
        self._uniform_data["time"] = time_float
        self._uniform_data["mouse"] = mouse_pos
        self._canvas.request_draw(self._draw_frame)
        frame = self._canvas.draw()
        return frame


if __name__ == "__main__":
    shader = Shadertoy(
        """
    fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
        let uv = frag_coord / i_resolution.xy;

        if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
            return vec4<f32>(textureSample(i_channel0, sampler0, uv));
        }else{
            return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
        }

    }
    """
    )
    shader.show()

class RenderPass():
    """
    Base class for renderpass in a Shadertoy.
    Parameters:
        parent(Shadertoy): the main Shadertoy class of which this renderpass is part of.
        code (str): Shadercode for this buffer.
        shader_type(str): either "wgsl" or "glsl" can also be "auto" - which then gets solved by a regular expression, we should be able to match differnt renderpasses... Defaults to glsl
        inputs (list): A list of :class:`ShadertoyChannel` objects. Each pass supports up to 4 inputs/channels. If a channel is dected in the code but none provided, will be sampling a black texture.
    """
    def __init__(self, parent:Shadertoy,  code:str, shader_type:str="glsl",inputs=[]) -> None:
        self.parent = parent
        self._shader_type = shader_type
        self._shader_code = code
        self.channels = self._attach_inputs(inputs)

    def _attach_inputs(self, inputs:list) -> list:
        if len(inputs) > 4:
            raise ValueError("Only 4 inputs supported")
        channels = []
        channel_pattern = re.compile(
            r"(?:iChannel|i_channel)(\d+)"
        )
        detected_channels = channel_pattern.findall(self.shader_code)


        return channels

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

class ImageRenderPass(RenderPass):
    """
    The Image RenderPass of a Shadertoy.
    """
    pass

class BufferRenderpass(RenderPass):
    """
    The Buffer A-D RenderPass of a Shadertoy.
    Parameters:
        buffer_idx (str): one of "A", "B", "C" or "D". Required.
    """
    def __init__(self, buffer_idx, **kwargs):
        super().__init__(**kwargs)

    pass

class CubemapRenderpass(RenderPass):
    """
    The Cube A RenderPass of a Shadertoy.
    this has slightly different headers see: https://shadertoyunofficial.wordpress.com/2016/07/20/special-shadertoy-features/
    """
    pass #TODO at a later date


class SoundRenderPass(RenderPass):
    """
    The Sound RenderPass of a Shadertoy.
    sound is rendered to a buffer at the start and then played back. There is no interactivity....
    """
    pass #TODO at a later date
