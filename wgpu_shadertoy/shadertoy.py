import collections
import ctypes
import os
import time

import wgpu
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas
from wgpu.gui.offscreen import run as run_offscreen

from .api import shader_args_from_json, shadertoy_from_id
from .passes import BufferRenderPass, ImageRenderPass, RenderPass


class UniformArray:
    """Convenience class to create a uniform array.

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
        resolution (tuple): The resolution of the shadertoy in (width, height). Defaults to (800, 450).
        shader_type (str): Can be "wgsl" or "glsl". On any other value, it will be automatically detected from shader_code. Default is "auto".
        offscreen (bool): Whether to render offscreen. Default is False.
        inputs (list): A list of :class:`ShadertoyChannel` objects. Supports up to 4 inputs. Defaults to sampling a black texture.
        title (str): The title of the window. Defaults to "Shadertoy".
        complete (bool): Whether the shader is complete. Unsupported renderpasses or inputs will set this to False. Default is True.

    The shader code must contain a entry point function:

    WGSL: ``fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{}`` <br>
    GLSL: ``void mainImage(out vec4 fragColor, in vec2 fragCoord){}``

    It has a parameter ``frag_coord`` which is the current pixel coordinate (in range 0..resolution, origin is bottom-left),
    and it must return a vec4<f32> color (for GLSL, it's the ``out vec4 fragColor`` parameter), which is the color of the pixel at that coordinate.

    some built-in uniforms are available in the shader:

    * ``i_mouse``: the mouse position in pixels
    * ``i_date``: the current date and time as a vec4 (year, month, day, seconds)
    * ``i_resolution``: the resolution of the shadertoy
    * ``i_time``: the global time in seconds
    * ``i_time_delta``: the time since last frame in seconds
    * ``i_frame``: the frame number
    * ``i_framerate``: the number of frames rendered in the last second.

    For GLSL, these uniforms are ``iTime``, ``iTimeDelta``, ``iFrame``, ``iResolution``, ``iMouse``, ``iDate`` and ``iFrameRate``,
    the entry point function is ``mainImage``, so you can use the shader code copied from shadertoy website without making any changes.
    """

    # todo: add remaining built-in variables (i_channel_time)
    # todo: support multiple render passes (`i_channel0`, `i_channel1`, etc.)

    def __init__(
        self,
        shader_code: str,
        common: str = "",
        resolution=(800, 450),
        shader_type="auto",
        offscreen=None,
        inputs=[],
        buffers: list[BufferRenderPass] = [],
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
        self._uniform_data["resolution"] = (*resolution, 1)
        self._shader_type = shader_type.lower()

        # if no explicit offscreen option was given
        # inherit wgpu-py force offscreen option
        if offscreen is None and os.environ.get("WGPU_FORCE_OFFSCREEN") == "true":
            offscreen = True
        self._offscreen = offscreen

        self.title = title
        self.complete = complete
        if not self.complete:
            self.title += " (incomplete)"

        device_features = []
        if buffers:
            device_features.append(wgpu.FeatureName.float32_filterable)
        self._device = self._request_device(device_features)

        self._prepare_canvas()
        self._bind_events()

        # setting up the renderpasses, inputs to the main class get handed to the .image pass
        self.image = ImageRenderPass(
            code=shader_code, shader_type=shader_type, inputs=inputs
        )

        # register all the buffers
        self.buffers: dict[str, BufferRenderPass] = {}
        if len(buffers) > 4:
            raise ValueError("Only 4 buffers are supported.")
        for buf in buffers:
            self.buffers[buf.buffer_idx] = buf

        # # finish the initialization by setting .main _prepare_render
        for rp in self.renderpasses:
            rp.main = self
        # only after main has been set, we can _prepare_render(), there can be complex cross-references
        # TODO: maybe just do a global for main?
        for rp in self.renderpasses:
            rp._prepare_render()

    @property
    def resolution(self):
        """The resolution of the shadertoy as a tuple (width, height) in pixels."""
        return tuple(self._uniform_data["resolution"])[:2]

    # TODO: this could be part of __init__
    @property
    def renderpasses(self) -> list[RenderPass]:
        """returns a list of active renderpasses, in render order."""
        if not hasattr(self, "_renderpasses"):
            self._renderpasses = []
            for buf in self.buffers.values():
                if buf:
                    self._renderpasses.append(buf)
            # TODO: where will cube and sound go?
            self._renderpasses.append(self.image)
        return self._renderpasses

    def _request_device(self, features) -> wgpu.GPUDevice:
        """
        returns the _global_device if no features are required
        otherwise requests a new device with the required features
        this logic is needed to pass unit tests due to how we run examples.
        Might be deprecated in the future, ref: https://github.com/pygfx/wgpu-py/pull/517
        """
        if not features:
            return wgpu.utils.get_default_device()

        return wgpu.gpu.request_adapter_sync(
            power_preference="high-performance"
        ).request_device_sync(required_features=features)

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

    def _prepare_canvas(self):
        if self._offscreen:
            self._canvas = OffscreenCanvas(
                title=self.title, size=self.resolution, max_fps=60
            )
        else:
            self._canvas = WgpuCanvas(
                title=self.title, size=self.resolution, max_fps=60
            )

        self._present_context = self._canvas.get_context()

        # We use non srgb variants, because we want to let the shader fully control the color-space.
        # Defaults usually return the srgb variant, but a non srgb option is usually available
        # comparable: https://docs.rs/wgpu/latest/wgpu/enum.TextureFormat.html#method.remove_srgb_suffix
        self._format = self._present_context.get_preferred_format(
            adapter=self._device.adapter
        ).removesuffix("-srgb")

        self._present_context.configure(device=self._device, format=self._format)

    def _bind_events(self):
        def on_resize(event):
            w, h = event["width"], event["height"]
            self._uniform_data["resolution"] = (w, h, 1)
            for buf in self.buffers.values():
                # TODO: do we want to call this every single time or only when the resize is done?
                # render loop is suspended during any window interaction anyway - will be fixed with rendercanvas: https://github.com/pygfx/rendercanvas/issues/69
                buf.resize_buffer()

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

        current_time = time.time()
        time_struct = time.localtime(current_time)
        fractional_seconds = current_time % 1

        self._uniform_data["date"] = (
            float(time_struct.tm_year),
            float(time_struct.tm_mon - 1),
            float(time_struct.tm_mday),
            time_struct.tm_hour * 3600
            + time_struct.tm_min * 60
            + time_struct.tm_sec
            + fractional_seconds,
        )

        self._uniform_data["frame"] = self._frame
        self._frame += 1

    def _draw_frame(self):
        # Update uniform buffer
        if not self._offscreen:
            self._update()

        # record all renderpasses into encoders
        render_encoders = []
        for rpass in self.renderpasses:
            pass_encoder_buffer = rpass.draw()
            render_encoders.append(pass_encoder_buffer)

        self._device.queue.submit(render_encoders)
        self._canvas.request_draw()

    def show(self):
        self._canvas.request_draw(self._draw_frame)
        if self._offscreen:
            run_offscreen()
        else:
            run()

    def snapshot(
        self,
        time_float: float = 0.0,
        time_delta: float = 0.167,
        frame: int = 0,
        framerate: int = 60.0,
        mouse_pos: tuple = (0.0, 0.0, 0.0, 0.0),
        date: tuple = (0.0, 0.0, 0.0, 0.0),
    ) -> memoryview:
        """
        Returns an image of the specified time. (Only available when ``offscreen=True``), you can set the uniforms manually via the parameters.
        Snapshots will be saved in the channel order of self._format.

        Parameters:
            time_float (float): The time to snapshot. It essentially sets ``i_time`` to a specific number. (Default is 0.0)
            time_delta (float): Value for ``i_time_delta`` uniform. (Default is 0.167)
            frame (int): The frame number for ``i_frame`` uniform. (Default is 0)
            framerate (float): The framerate number for ``i_framerate``, only changes the value passed to the uniform. (Default is 60.0)
            mouse_pos (tuple(float)): The mouse position in pixels in the snapshot. It essentially sets ``i_mouse`` to a 4-tuple. (Default is (0.0,0.0,0.0,0.0))
            date (tuple(float)): The 4-tuple for ``i_date`` in year, months, day, seconds. (Default is (0.0,0.0,0.0,0.0))
        Returns:
            frame (memoryview): snapshot with transparency. This object can be converted to a numpy array (without copying data)
        using ``np.asarray(arr)``
        """
        if not self._offscreen:
            raise NotImplementedError("Snapshot is only available in offscreen mode.")

        self._uniform_data["time"] = time_float
        self._uniform_data["time_delta"] = time_delta
        self._uniform_data["frame"] = frame
        self._uniform_data["framerate"] = framerate
        self._uniform_data["mouse"] = mouse_pos
        self._uniform_data["date"] = date
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
