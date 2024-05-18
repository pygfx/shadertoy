import numpy as np
import wgpu


class ShadertoyChannel:
    """
    ShadertoyChannel Base class. If nothing is provided, it defaults to a 8x8 black texture.
    Parameters:
        ctype (str): channeltype, can be "texture", "buffer", "video", "webcam", "music", "mic", "keyboard", "cubemap", "volume"; default assumes texture.
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
        self.args = args
        self.kwargs = kwargs

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
    def sampler_settings(self):
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
    def channel_res(self) -> tuple:
        """iChannelResolution[N] information for the uniform. Tuple of (width, height, 1, -99)"""
        raise NotImplementedError("likely implemented for ChannelTexture")

    def create_texture(self, device):
        raise NotImplementedError(
            "This method should likely be implemented in the subclass - but maybe it's all the same? TODO: check later!"
        )

    def binding_layout(self, offset=0):
        # TODO: figure out how offset works when we have multiple passes
        texture_binding = (2 * self.channel_idx) + 1
        sampler_binding = 2 * (self.channel_idx + 1)

        return [
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

    def bind_groups_layout_entries(self, texture_view, sampler, offset=0):
        # TODO maybe refactor this all into a prepare bindings method?
        texture_binding = (2 * self.channel_idx) + 1
        sampler_binding = 2 * (self.channel_idx + 1)
        return [
            {
                "binding": texture_binding,
                "resource": texture_view,
            },
            {
                "binding": sampler_binding,
                "resource": sampler,
            },
        ]

    def header_glsl(self, input_idx=0):
        """
        GLSL code snippet added to the compatibilty header for Shadertoy inputs.
        """
        binding_id = (2 * input_idx) + 1
        sampler_id = 2 * (input_idx + 1)
        return f"""
        layout(binding = {binding_id}) uniform texture2D i_channel{input_idx};
        layout(binding = {sampler_id}) uniform sampler sampler{input_idx};
        #define iChannel{input_idx} sampler2D(i_channel{input_idx}, sampler{input_idx})
        """

    def header_wgsl(self, input_idx=0):
        """
        WGSL code snippet added to the compatibilty header for Shadertoy inputs.
        """
        binding_id = (2 * input_idx) + 1
        sampler_id = 2 * (input_idx + 1)
        return f"""
        @group(0) @binding{binding_id}
        var i_channel{input_idx}: texture_2d<f32>;
        @group(0) @binding({sampler_id})
        var sampler{input_idx}: sampler;
        """

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
        buffer_or_pass (str|BufferRenderPass): The buffer index, can be one oif ("A", "B", "C", "D"), or the buffer itself.
        parent (RenderPass): The main renderpass this buffer is attached to. (optional in the init, but should be set later)
        **kwargs for sampler settings.
    """

    def __init__(self, buffer, parent=None, **kwargs):
        super().__init__(**kwargs)
        self.buffer_idx = buffer  # A,B,C or D?
        if parent is not None:
            self._parent = parent

    def create_texture(self, device):
        texture = device.create_texture(
            size=self.parent.size,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        return texture


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

        self.size = self.data.shape  # (rows, columns, channels)
        self.texture_size = (
            self.data.shape[1],
            self.data.shape[0],
            1,
        )  # orientation change (columns, rows, 1)
        self.bytes_per_pixel = (
            self.data.nbytes // self.data.shape[1] // self.data.shape[0]
        )
        vflip = kwargs.pop("vflip", True)
        if vflip in ("true", True):
            vflip = True
            self.data = np.ascontiguousarray(self.data[::-1, :, :])

    @property
    def channel_res(self) -> tuple:
        return (
            self.size[1],
            self.size[0],
            1,
            -99,
        )  # (width, height, pixel_aspect=1, padding=-99)

    def create_texture(self, device):
        texture = device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        return texture


class ShadertoyChannelCubemap(ShadertoyChannel):
    pass


class ShadertoyChannelVolume(ShadertoyChannel):
    pass


class ShadertoyChannelVideo(ShadertoyChannel):
    pass


class ShadertoyChannelMusic(ShadertoyChannel):
    pass
