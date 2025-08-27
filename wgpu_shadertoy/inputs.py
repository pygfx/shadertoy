from typing import Tuple

import numpy as np
import wgpu
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .passes import RenderPass


# https://developer.mozilla.org/en-US/docs/Web/API/UI_Events/Keyboard_event_key_values looks like VK_* match
KEY_MAP = {
    "Alt": 18,
    "CapsLock": 20,
    "Control": 17,
    "Meta": 91,
    "NumLock": 144,
    "ScrollLock": 145,
    "Shift": 16,
    "Enter": 13,
    "Tab": 9,
    " ": 32, # space
    "ArrowDown": 40,
    "ArrowLeft": 37,
    "ArrowRight": 39,
    "ArrowUp": 38,
    "End": 35,
    "Home": 36,
    "PageDown": 34,
    "PageUp": 33,
    "Backspace": 8,
    "Delete": 46,
    "Escape": 27,
    "Pause": 19,
    "F1": 112,
    "F2": 113,
    "F3": 114,
    "F4": 115,
    "F5": 116,
    "F6": 117,
    "F7": 118,
    "F8": 119,
    "F9": 120,
    "F10": 121,
    "F11": 122,
    "F12": 123,
}

class ShadertoyChannel:
    """
    ShadertoyChannel Base class. If nothing is provided, it defaults to a 8x8 black texture.
    Parameters:
        ctype (str): channeltype, can be "texture", "buffer", "video", "webcam", "music", "mic", "keyboard", "cubemap", "volume"; default assumes texture.
        channel_idx (int): The channel index, can be one of (0, 1, 2, 3). Default is None. It will be set by the parent renderpass.
        **kwargs: Additional arguments for the sampler:
        wrap (str): The wrap mode, can be one of ("clamp-to-edge", "repeat", "clamp"). Default is "clamp-to-edge".
    """

    def __init__(self, *args, ctype=None, channel_idx=None, **kwargs):
        self.ctype = ctype
        if channel_idx is None:
            channel_idx = kwargs.pop("channel_idx", None)
        self._channel_idx = channel_idx
        self.args = args
        self.kwargs = kwargs
        self.dynamic: bool = False  # is this still needed? should it be private?
        self._parent = kwargs.get("parent", None)

    @property
    def sampler_settings(self) -> dict:
        """
        Sampler settings for this channel. Wrap currently supported. Filter not yet.
        """
        settings = {}
        wrap = self.kwargs.get("wrap", "clamp-to-edge")
        # "warp", "clamp" or "repeat" is what we should expect on Shadertoy
        if wrap.startswith("clamp"):
            wrap = "clamp-to-edge"
        settings["address_mode_u"] = wrap
        settings["address_mode_v"] = wrap
        # we don't do 3D textures yet, but I guess setting this too is fine.
        settings["address_mode_w"] = wrap

        filter = self.kwargs.get("filter", "linear")
        # wgpu.FilterMode is either "linear" or "nearest", "mipmap" requires special attention
        if filter not in ("linear", "nearest"):
            filter = "linear"  # work around to avoid mipmap mode for now

        # both min and mag will use the same filter.
        settings["mag_filter"] = filter
        settings["min_filter"] = filter
        return settings

    @property
    def parent(self):
        """
        Parent renderpass of this channel.
        """
        if self._parent is None:
            raise AttributeError("Parent not set.")
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent

    @property
    def channel_idx(self) -> int:
        if self._channel_idx is None:
            raise AttributeError("Channel index not set.")
        return self._channel_idx

    @channel_idx.setter
    def channel_idx(self, idx=int):
        if idx not in (0, 1, 2, 3):
            raise ValueError("Channel index must be in [0,1,2,3]")
        self._channel_idx = idx

    @property
    def texture_binding(self) -> int:
        return (2 * self.channel_idx) + 1

    @property
    def sampler_binding(self) -> int:
        return 2 * (self.channel_idx + 1)

    @property
    def channel_res(self) -> Tuple[int, int, int, int]:
        """
        Tuple of (width, height, pixel_aspect=1, padding=-99)
        """
        return (self.size[1], self.size[0], 1, -99)

    @property
    def size(self) -> Tuple:  # what shape tho?
        """
        Size of the texture.
        """
        return self.data.shape

    def infer_subclass(self, *args_, **kwargs_):
        """
        Returns an instance of the relevant subclass with the provided arguments.
        """
        # TODO: automatically infer from the provided data/file/link/name or code.
        args = self.args + args_
        kwargs = {**self.kwargs, **kwargs_}
        if self.ctype is None or not hasattr(self, "ctype"):
            raise NotImplementedError("Can't dynamically infer the ctype yet")
        if self.ctype == "texture":
            return ShadertoyChannelTexture(*args, **kwargs)
        elif self.ctype == "buffer":
            return ShadertoyChannelBuffer(*args, **kwargs)
        elif self.ctype == "keyboard":
            return ShadertoyChannelKeyboard(*args, **kwargs)
        else:
            raise NotImplementedError(f"Doesn't support {self.ctype=} yet")

    # TODO: can this be avoided?
    def _binding_layout(self):
        return [
            {
                "binding": self.texture_binding,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                },
            },
            {
                "binding": self.sampler_binding,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.filtering},
            },
        ]

    def _bind_groups_layout_entries(self, texture_view, sampler) -> list:
        # TODO maybe refactor this all into a prepare bindings method?
        return [
            {
                "binding": self.texture_binding,
                "resource": texture_view,
            },
            {
                "binding": self.sampler_binding,
                "resource": sampler,
            },
        ]

    def make_header(self, shader_type: str) -> str:
        """
        Constructs the glsl or wgsl code snippet that for the sampler and texture bindings.
        """
        # does this fallback ever happen?
        if not shader_type:
            shader_type = self.parent.shader_type
        shader_type = shader_type.lower()

        input_idx = self.channel_idx
        binding_id = self.texture_binding
        sampler_id = self.sampler_binding
        if shader_type == "glsl":
            return f"""
            layout(binding = {binding_id}) uniform texture2D si_channel{input_idx};
            layout(binding = {sampler_id}) uniform sampler sampler{input_idx};
            
            #define iChannel{input_idx} sampler2D(si_channel{input_idx}, sampler{input_idx})
            """
        elif shader_type == "wgsl":
            return f"""
            @group(0) @binding({binding_id})
            var i_channel{input_idx}: texture_2d<f32>;
            @group(0) @binding({sampler_id})
            var sampler{input_idx}: sampler;
            """
        else:
            raise ValueError(f"Unknown shader type: {shader_type}")

    def __repr__(self):
        """
        Convenience method to get a representation of this object for debugging.
        """
        if hasattr(self, "data"):
            data_repr = {
                "repr": self.data.__repr__(),
                "shape": self.data.shape,
                "strides": self.data.strides,
                "nbytes": self.data.nbytes,
            }
        else:
            data_repr = None
        class_repr = {k: v for k, v in self.__dict__.items() if k != "data"}
        class_repr["data"] = data_repr
        class_repr["class"] = self.__class__
        return repr(class_repr)


# "Misc" input tab
class ShadertoyChannelKeyboard(ShadertoyChannel):
    # isn't this basically a texture/video??
    # can we have a GPU buffer and then just write it to texture?
    # do we only update on keypresses or do we do it every frame (can you do multiple keypressese faster than a frame?)
    # ref: https://www.shadertoy.com/view/lsXGzf
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.format = wgpu.TextureFormat.r8unorm #or r8sint?
        self.dynamic = True  # could be named "needs_update" to be more clear
        self.vflip = True #always true but we handle that manually, so don't really need the var!

        # when we get this via the ._infer_subclass() method - we might already have a parent and can register the events now!
        if self._parent is not None:
            self.register_events()
            
    # do we have to redo both parts?
    @property
    def parent(self) -> "RenderPass":
        """
        Parent renderpass of this channel.
        """
        if self._parent is None:
            raise AttributeError("Parent not set.")
        return self._parent

    @parent.setter
    def parent(self, parent):
        self._parent = parent
        # register the events here?
        self.register_events()


    @property
    def data(self):
        """
        Keyboard state is shared across all renderpasses that might use this input channel.
        """
        # TODO: maybe we should have some kind of _shared object for this and more
        if not hasattr(self.parent.main, "_keyboard_state"):
            self.parent.main._keyboard_state = np.zeros((3, 256), dtype=np.uint8)  # 3 rows, 256 keys and only one channel
        return self.parent.main._keyboard_state
    
    @property
    def texture(self):
        # texture is also shared 
        if not hasattr(self.parent.main, "_keyboard_texture"):
            self.parent.main._keyboard_texture = self.parent.main._device.create_texture(
                size=(256, 3, 1),  # note it's columns, rows here
                format=self.format,
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
        return self.parent.main._keyboard_texture

    def register_events(self):
        """
        Register the keyboard event handlers only when we need them.
        """
        # TODO: if events already exist just exit...
        # print(self.parent.main._canvas._events._event_handlers)
        if self.parent.main._canvas._events._event_handlers.get("key_down") and self.parent.main._canvas._events._event_handlers.get("key_up"):
            # print("Keyboard events already registered, skipping...")
            return

        def on_key_down(event):
            try:
                key_code = KEY_MAP.get(event["key"]) or ord(event["key"].upper()) # we only want to evaluate ord() if there is no mapping!
                assert key_code < 256, f"Key {event['key']} code {key_code} is out of range for keyboard channel."
            except TypeError:
                print(f"Key event error: {event['key']}")
                key_code = 0
            self.data[0, key_code] = 255
            self.data[1, key_code] = 255 # this only stays for a little bit of time?
            self.data[2, key_code] = 255 if self.data[2, key_code] == 0 else 0 # toggle pressed state
            self.dynamic = 2  # basically tell it to update for next frame and one after that!

        def on_key_up(event):
            try:
                key_code = KEY_MAP.get(event["key"]) or ord(event["key"].upper())
                assert key_code < 256
            except TypeError:
                key_code = 0
            self.data[0, key_code] = 0 # up action only triggers the "state" row
            self.dynamic = True

        self.parent.main._canvas.add_event_handler(on_key_down, "key_down")
        self.parent.main._canvas.add_event_handler(on_key_up, "key_up")



    # copied from ShadertoyChannelTexture, changed sizes (maybe it could be moved to the base class?)
    def bind_texture(self, device: wgpu.GPUDevice) -> Tuple[list, list]:
        """
        prepares the texture and sampler. Returns it's binding layouts and bindgroup layout entries
        """
        # this gets called during the draw too... so every single time (via _setup_renderpipeline!)

        # texture bindings are unique per pass, using the keyboard twice will lead to a reuse but should hopefully still work.
        binding_layout = self._binding_layout()

        # TODO: this could also be cached...
        texture_view = self.texture.create_view()
        
        # sampler can actually be unique per channel due to filtering and wrap settings.
        sampler = device.create_sampler(**self.sampler_settings)

        bind_groups_layout_entry = self._bind_groups_layout_entries(
            texture_view, sampler
        )

        return binding_layout, bind_groups_layout_entry

    # TODO: this should only be called once per frame and hence might need to be part of the main.update function...
    def update(self, device: wgpu.GPUDevice):
        # to be called just before the draw call for this pass:
        device.queue.write_texture(
            destination={
                "texture": self.texture,
            },
            data=np.ascontiguousarray(self.data),
            data_layout={
                "bytes_per_row": 256, # int8 texture of 256
                "rows_per_image": 3,
            },
            size=self.texture.size,
        )
        self.data[1, :] = np.zeros(256, dtype=np.uint8)  # reset the second row to 0s after we uploaded that data (could be an issue if we reuse this channel...)

        # because we need to essentially draw one more time
        self.dynamic -= 1

class ShadertoyChannelWebcam(ShadertoyChannel):
    pass


class ShadertoyChannelMicrophone(ShadertoyChannel):
    pass


class ShadertoyChannelSoundcloud(ShadertoyChannel):
    pass


class ShadertoyChannelBuffer(ShadertoyChannel):
    def __init__(self, buffer, **kwargs):
        super().__init__(**kwargs)

        if isinstance(buffer, str):
            # when the user gives a string, we don't have the associated buffer renderpass yet
            self.buffer_idx = buffer.lower()[-1]
            self._renderpass = None
        else:  # (assume BufferPass instance?)
            self._renderpass = buffer
            self.buffer_idx = buffer.buffer_idx

        # mark that this channel needs to be updated every frame
        self.dynamic = True

    @property
    def size(self):
        """
        texture size of the front texture
        """
        return self.renderpass.texture_front.size

    @property
    def renderpass(self):  # -> BufferRenderPass:
        if self._renderpass is None:
            self._renderpass = self.parent.main.buffers[self.buffer_idx]
        return self._renderpass
    
    def update(self, device: wgpu.GPUDevice):
        # TODO: buffer passes get updated through a combination of the draw function swapping textures
        # and the draw function also calling _setup_renderpipeline() to get the newer bindings...
        # this seems wasteful and should be improved!

        # not the solution as we call it too often from here...
        # self.renderpass._setup_renderpipeline()
        pass # to avoid warnings here.

    def bind_texture(self, device: wgpu.GPUDevice) -> Tuple[list, list]:
        """
        returns a tuple of binding_layout and binding_groups_layout_entries
        takes the texture form `front` the buffer renderpass (last frame)
        """
        binding_layout = self._binding_layout()
        texture: wgpu.GPUTexture = self.renderpass.texture_front
        texture_view = texture.create_view(usage=wgpu.TextureUsage.TEXTURE_BINDING)
        sampler = device.create_sampler(**self.sampler_settings)
        bind_groups_layout_entry = self._bind_groups_layout_entries(
            texture_view, sampler
        )
        return binding_layout, bind_groups_layout_entry


class ShadertoyChannelCubemapA(ShadertoyChannel):
    pass


# other tabs
class ShadertoyChannelTexture(ShadertoyChannel):
    """
    Represents a Shadertoy texture input channel.
    Parameters:
        data (array-like): Of shape (width, height, channels), will be converted to numpy array. Default is a 8x8 black texture.
        **kwargs: Additional arguments for the sampler:
        wrap (str): The wrap mode, can be one of ("clamp-to-edge", "repeat", "clamp"). Default is "clamp-to-edge".
        vflip (str or bool): Whether to flip the texture vertically. Can be one of ("true", "false", True, False). Default is True.
    """

    # TODO: can we inherent the sampler args part of the docstring?
    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)
        if data is not None:
            self.data = np.ascontiguousarray(data)
        else:
            self.data = np.zeros((8, 8, 4), dtype=np.uint8)

        # if channel dimension is missing, it's a greyscale texture
        if len(self.data.shape) == 2:
            self.data = np.reshape(self.data, self.data.shape + (1,))
        # greyscale textures become just red while green and blue remain 0s
        if self.data.shape[2] == 1:
            self.data = np.stack(
                [
                    self.data[:, :, 0],
                    np.zeros_like(self.data[:, :, 0]),
                    np.zeros_like(self.data[:, :, 0]),
                ],
                axis=-1,
            )
        # if alpha channel is not given, it's filled with max value (255)
        if self.data.shape[2] == 3:
            self.data = np.concatenate(
                [self.data, np.full(self.data.shape[:2] + (1,), 255, dtype=np.uint8)],
                axis=2,
            )

        # orientation change (columns, rows, 1)
        self.texture_size = (self.data.shape[1], self.data.shape[0], 1)

        vflip = kwargs.pop("vflip", True)
        if vflip in ("true", True):
            vflip = True
            self.data = np.ascontiguousarray(self.data[::-1, :, :])

    def bind_texture(self, device: wgpu.GPUDevice) -> Tuple[list, list]:
        """
        prepares the texture and sampler. Returns it's binding layouts and bindgroup layout entries
        """

        binding_layout = self._binding_layout()
        texture = device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
        )

        texture_view = texture.create_view()
        device.queue.write_texture(
            destination={
                "texture": texture,
            },
            data=self.data,
            data_layout={
                "bytes_per_row": self.data.strides[0],  # multiple of 256
                "rows_per_image": self.size[0],
            },
            size=texture.size,
        )

        sampler = device.create_sampler(**self.sampler_settings)

        bind_groups_layout_entry = self._bind_groups_layout_entries(
            texture_view, sampler
        )

        return binding_layout, bind_groups_layout_entry


class ShadertoyChannelCubemap(ShadertoyChannel):
    pass


class ShadertoyChannelVolume(ShadertoyChannel):
    pass


class ShadertoyChannelVideo(ShadertoyChannel):
    pass


class ShadertoyChannelMusic(ShadertoyChannel):
    pass
