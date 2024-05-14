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
        self._channel_idx = channel_idx
        self.args = args
        self.kwargs = kwargs

    def infer_subclass(self, *args_, **kwargs_):
        """
        Return the relevant subclass, instantiated with the provided arguments.
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
    def channel_idx(self) -> int:
        if self._channel_idx is None:
            raise AttributeError("Channel index not set.")
        return self._channel_idx

    @channel_idx.setter
    def channel_idx(self, idx=int):
        if idx not in (0, 1, 2, 3):
            raise ValueError("Channel index must be in [0,1,2,3]")
        self._channel_idx = idx

    # TODO: where do we get self.size in the base class from? else this should pass instead?
    @property
    def channel_res(self):
        return (
            self.size[1],
            self.size[0],
            1,
            -99,
        )  # (width, height, pixel_aspect=1, padding=-99)

    def header_glsl(self, input_idx=0):
        """
        GLSL code that provides compatibility with Shadertoys input channels.
        """
        binding_id = (2 * input_idx) + 1
        sampler_id = 2 * (input_idx + 1)
        f"""
        layout(binding = {binding_id}) uniform texture2D i_channel{input_idx};
        layout(binding = {sampler_id}) uniform sampler sampler{input_idx};
        #define iChannel{input_idx} sampler2D(i_channel{input_idx}, sampler{input_idx})
        """

    def header_wgsl(self, input_idx=0):
        """
        WGSL code that provides compatibility with WGLS translated Shadertoy inputs.
        """
        binding_id = (2 * input_idx) + 1
        sampler_id = 2 * (input_idx + 1)
        f"""
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
        buffer (str): The buffer index, can be one of ("A", "B", "C", "D").
        main (Shadertoy): The main Shadertoy class this buffer is attached to.
        code (str): The shadercode of this buffer, will be handed to the main Shadertoy class (optional).
        inputs (list): List of ShadertoyChannel objects that this buffer uses. (can be itself?)
    """

    def __init__(self, buffer, code="", inputs=None, main=None, **kwargs):
        self.buffer_idx = buffer  # A,B,C or D?
        self.main = (
            main  # the main image class it's attached to? not strictly the parent.
        )
        if not code:
            self.code = main.buffer.get(buffer, "")
        else:
            self.code = code
            main.buffer[buffer] = code  # set like this??

        # TODO: reuse the code from the main class?
        self.inputs = inputs


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

    def binding_layout(self, binding_idx, sampler_binding):
        return [
            {
                "binding": binding_idx,
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


class ShadertoyChannelCubemap(ShadertoyChannel):
    pass


class ShadertoyChannelVolume(ShadertoyChannel):
    pass


class ShadertoyChannelVideo(ShadertoyChannel):
    pass


class ShadertoyChannelMusic(ShadertoyChannel):
    pass
