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

    # TODO: infer ctype from provided data/link/file or specified "cytype" argument, can we return the proper class?
    # TODO: sampler filter modes: nearest, linear, mipmap (figure out what they mean in wgpu).
    def __init__(self, **kwargs):
        """
        Superclass inits for the sampling args.
        """
        self.sampler_settings = {}
        wrap = kwargs.pop("wrap", "clamp-to-edge")
        if wrap.startswith("clamp"):
            wrap = "clamp-to-edge"
        self.sampler_settings["address_mode_u"] = wrap
        self.sampler_settings["address_mode_v"] = wrap
        self.sampler_settings["address_mode_w"] = wrap

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
        data_repr = {
            "repr": self.data.__repr__(),
            "shape": self.data.shape,
            "strides": self.data.strides,
            "nbytes": self.data.nbytes,
        }
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
    Shadertoy Buffer input, takes the fragment code and it's own channel inputs.
    Renders to a buffer, which the main shader then uses as a texture.
    """

    def __init__(self, code="", inputs=None):
        self.code = code
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
