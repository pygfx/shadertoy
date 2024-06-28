from typing import Tuple

import numpy as np
import wgpu
import wgpu.structs


class ShadertoyChannel:
    """
    ShadertoyChannel Base class. If nothing is provided, it defaults to a 8x8 black texture.
    Parameters:
        ctype (str): channeltype, can be "texture", "buffer", "video", "webcam", "music", "mic", "keyboard", "cubemap", "volume"; default assumes texture.
        channel_idx (int): The channel index, can be one of (0, 1, 2, 3). Default is None. It will be set by the parent renderpass.
        **kwargs: Additional arguments for the sampler:
        wrap (str): The wrap mode, can be one of ("clamp-to-edge", "repeat", "clamp"). Default is "clamp-to-edge".
    """

    # TODO: infer ctype from provided data/link/file if ctype is not provided.
    # TODO: sampler filter modes: nearest, linear, mipmap (figure out what they mean in wgpu).
    def __init__(self, *args, ctype=None, channel_idx=None, **kwargs):
        self.ctype = ctype
        if channel_idx is None:
            channel_idx = kwargs.pop("channel_idx", None)
        self._channel_idx = channel_idx
        self.args = args  # actually reduddant?
        self.kwargs = kwargs
        self.dynamic = False

    def infer_subclass(self, *args_, **kwargs_):
        """
        Return the relevant subclass, instantiated with the provided arguments.
        TODO: automatically infer it from the provided data/file/link or code.
        """
        args = self.args + args_
        kwargs = {**self.kwargs, **kwargs_}
        if self.ctype is None or not hasattr(self, "ctype"):
            raise NotImplementedError("Can't dynamically infer the ctype yet")
        if self.ctype == "texture":
            return ShadertoyChannelTexture(
                *args, channel_index=self._channel_idx, **kwargs
            )
        elif self.ctype == "buffer":
            return ShadertoyChannelBuffer(
                *args, channel_index=self._channel_idx, **kwargs
            )

    @property
    def sampler_settings(self) -> dict:
        """
        Sampler settings for this channel. Wrap currently supported. Filter not yet.
        """
        sampler_settings = {}
        wrap = self.kwargs.get("wrap", "clamp-to-edge")
        if wrap.startswith("clamp"):
            wrap = "clamp-to-edge"
        sampler_settings["address_mode_u"] = wrap
        sampler_settings["address_mode_v"] = wrap
        sampler_settings["address_mode_w"] = wrap
        return sampler_settings

    @property
    def parent(self):
        # TODO: likely make a passes.py file to make typing possible -> RenderPass:
        """Parent of this input is a renderpass."""
        if not hasattr(self, "_parent"):
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
    def channel_res(self) -> Tuple[int]:
        return (
            self.size[1],
            self.size[0],
            1,
            -99,
        )  # (width, height, pixel_aspect=1, padding=-99)

    @property
    def size(self) -> tuple:  # tuple?
        return self.data.shape

    @property
    def bytes_per_pixel(self) -> int:
        return 4  # shortcut for speed?
        # usually is 4 for rgba8unorm or maybe use self.data.strides[1]?
        # print(self.data.shape, self.data.nbytes)
        bpp = self.data.nbytes // self.data.shape[1] // self.data.shape[0]

        return bpp

    def bind_texture(self, device: wgpu.GPUDevice) -> Tuple[list, list]:
        """
        returns a tuple of binding_layout and bing_groups_layout_entries.
        writes textures as well as creates the sampler.
        """
        raise NotImplementedError("This method should be implemented in the subclass.")

    def _binding_layout(self, offset=0):
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

    def _bind_groups_layout_entries(self, texture_view, sampler, offset=0) -> list:
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

    def get_header(self, shader_type: str = "") -> str:
        """
        GLSL or WGSL code snippet added to the compatibility header for Shadertoy inputs.
        """
        if not shader_type:
            shader_type = self.parent.shader_type
        shader_type = shader_type.lower()

        input_idx = self.channel_idx
        binding_id = self.texture_binding
        sampler_id = self.sampler_binding
        if shader_type == "glsl":
            return f"""
            layout(binding = {binding_id}) uniform texture2D i_channel{input_idx};
            layout(binding = {sampler_id}) uniform sampler sampler{input_idx};
            
            #define iChannel{input_idx} sampler2D(i_channel{input_idx}, sampler{input_idx})
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
    pass


class ShadertoyChannelWebcam(ShadertoyChannel):
    pass


class ShadertoyChannelMicrophone(ShadertoyChannel):
    pass


class ShadertoyChannelSoundcloud(ShadertoyChannel):
    pass


class ShadertoyChannelBuffer(ShadertoyChannel):
    """
    Shadertoy buffer texture input. The relevant code and renderpass resides in the main Shadertoy class.
    Parameters:
        buffer (str|`BufferRenderPass`): The buffer index, can be one of ("A", "B", "C", "D"), or the `BufferRenderPass` itself.
        parent (`RenderPass`): The main renderpass this buffer-texture is attached to. (optional in the init, but should be set later).
        **kwargs for sampler settings.
    """

    def __init__(self, buffer, parent=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(buffer, str):
            self.buffer_idx = buffer.lower()  # A,B,C or D?
            self._renderpass = None
        else:
            # TODO can we check for instance of BufferRenderPass?
            self._renderpass = buffer
            self.buffer_idx = buffer.buffer_idx

        if parent is not None:
            self._parent = parent
        self.dynamic = True

    @property
    def size(self):
        # width, height, 1, ?
        texture_size = self.renderpass.texture_size
        return (texture_size[1], texture_size[0], 1)

    @property
    def renderpass(self):  # -> BufferRenderPass:
        if self._renderpass is None:
            self._renderpass = self.parent.main.buffers[self.buffer_idx]
        return self._renderpass

    def bind_texture(self, device: wgpu.GPUDevice) -> Tuple[list, list]:
        """
        returns a tuple of binding_layout and bing_groups_layout_entries.
        takes the texture from the buffer and creates a new sampler.
        """
        binding_layout = self._binding_layout()
        texture = self.renderpass.texture
        texture_view = texture.create_view()
        sampler = device.create_sampler(**self.sampler_settings)
        # TODO: explore using auto layouts (pygfx/wgpu-py#500)
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

    def __init__(self, data=None, **kwargs):
        super().__init__(**kwargs)  # inherent the self.sampler_settings here?
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
        binding_layout = self._binding_layout()
        texture = device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        texture_view = texture.create_view()
        # typing missing in wgpu-py for queue
        # extract this to an update_texture method?
        # print(f"{self}, {self.data[0][0]=}")
        device.queue.write_texture(
            {
                "texture": texture,
                "origin": (0, 0, 0),
                "mip_level": 0,
            },
            self.data,
            {
                "offset": 0,
                "bytes_per_row": self.bytes_per_pixel * self.size[1],  # multiple of 256
                "rows_per_image": self.size[0],  # same is done internally
            },
            texture.size,
        )

        sampler = device.create_sampler(**self.sampler_settings)
        # TODO: explore using auto layouts (pygfx/wgpu-py#500)
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
