import collections
import ctypes
import os
import time

from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuCanvas as OffscreenCanvas
from wgpu.gui.offscreen import run as run_offscreen

from .api import shader_args_from_json, shadertoy_from_id
from .passes import BufferRenderPass, ImageRenderPass

# TODO: hacky solution, needs to be improved
_default_device = None


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
    """Provides a "screen pixel shader programming interface" similar to `shadertoy <https://www.shadertoy.com/>`.

    It helps you research and quickly build or test shaders using `WGSL` or `GLSL` via WGPU.

    Parameters:
        shader_code (str|ImageRenderPass): The shader code to use for the Image renderpass.
        common (str): The common shaderpass code gets executed before all other shaderpasses (buffers/image/sound). Defaults to empty string.
        buffers (dict(str|BufferRenderPass)): Codes for buffers A through D. Still requires to set buffer as channel input. Defaults to empty strings.
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
        },
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
        self._uniform_data["resolution"] = (*resolution, 1)
        self._shader_type = shader_type.lower()

        self.image = ImageRenderPass(
            main=self, code=shader_code, shader_type=shader_type, inputs=inputs
        )
        self.buffers = {"a": "", "b": "", "c": "", "d": ""}
        for k, v in buffers.items():
            k = k.lower()[-1]
            if k not in "abcd":
                raise ValueError(f"Invalid buffer key: {k}")
            if v == "":
                continue
            elif type(v) is BufferRenderPass:
                v.main = self
                v.buffer_idx = k
                self.buffers[k] = v
            elif not isinstance(v, str):
                raise ValueError(f"Invalid buffer value: {v=}")
            else:
                self.buffers[k] = BufferRenderPass(buffer_idx=k, code=v, main=self)

        # if no explicit offscreen option was given
        # inherit wgpu-py force offscreen option
        if offscreen is None and os.environ.get("WGPU_FORCE_OFFSCREEN") == "true":
            offscreen = True
        self._offscreen = offscreen

        if len(inputs) < 4:
            inputs.extend([None] * (4 - len(inputs)))
            # likely a better solution. But theoretically, someone could set one or more inputs but still mention a channel beyond that.

        self.title = title
        self.complete = complete

        if not self.complete:
            self.title += " (incomplete)"

        self._prepare_canvas()
        self._bind_events()
        # TODO: could this be part of the __init__ of each renderpass? (but we need the device)
        for rpass in (self.image, *self.buffers.values()):
            if rpass:  # skip None
                rpass.prepare_render(device=self._device)

    @property
    def resolution(self):
        """The resolution of the shadertoy as a tuple (width, height) in pixels."""
        return tuple(self._uniform_data["resolution"])[:2]

    @property
    def shader_code(self) -> str:
        """The shader code to use."""
        return self._shader_code

    # TODO: remove this redundant code snippet
    @property
    def shader_type(self) -> str:
        """
        The shader type of the main image renderpass.
        """
        return self.image.shader_type

    @property
    def _device(self):
        """
        copy and paste from wgpu.utils.device.get_default_device but with feature enabled
        we need to enable float32-filterable for buffer textures.
        """
        global _default_device

        if _default_device is None:
            import wgpu.backends.auto

            adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
            _default_device = adapter.request_device(
                required_features=["float32-filterable"]
            )
        return _default_device

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
        import wgpu.backends.auto

        if self._offscreen:
            self._canvas = OffscreenCanvas(
                title=self.title, size=self.resolution, max_fps=60
            )
        else:
            self._canvas = WgpuCanvas(
                title=self.title, size=self.resolution, max_fps=60
            )

        self._present_context = self._canvas.get_context()

        # We use "bgra8unorm" not "bgra8unorm-srgb" here because we want to let the shader fully control the color-space.
        self._present_context.configure(
            device=self._device, format=wgpu.TextureFormat.bgra8unorm
        )

    def _bind_events(self):
        def on_resize(event):
            w, h = event["width"], event["height"]
            self._uniform_data["resolution"] = (w, h, 1)
            for buf in self.buffers.values():
                if buf:
                    buf.resize(int(w), int(h))

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
            self.image._uniform_buffer,
            0,
            self._uniform_data.mem,
            0,
            self._uniform_data.nbytes,
        )

        for buf in self.buffers.values():
            if buf:  # checks if not None?
                buf.draw_buffer(self._device)
        self.image.draw_image(self._device, self._present_context)

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


# TODO: this code shouldn't be executed as a script anymore.
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
