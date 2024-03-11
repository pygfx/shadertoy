import numpy as np


class ShadertoyChannel:
    """
    Represents a shadertoy channel. It can be a texture.
    Parameters:
        data (array-like): Of shape (width, height, 4), will be converted to numpy array. Default is a 8x8 black texture.
        kind (str): The kind of channel. Can be one of ("texture"). More will be supported in the future
        **kwargs: Additional arguments for the sampler:
        wrap (str): The wrap mode, can be one of ("clamp-to-edge", "repeat", "clamp"). Default is "clamp-to-edge".
        vflip (str or bool): Whether to flip the texture vertically. Can be one of ("true", "false", True, False). Default is True.
    """

    # TODO: add cubemap/volume, buffer, webcam, video, audio, keyboard?

    def __init__(self, data=None, kind="texture", **kwargs):
        if kind != "texture":
            raise NotImplementedError("Only texture is supported for now.")
        if data is not None:
            self.data = np.ascontiguousarray(data)
        else:
            self.data = np.zeros((8, 8, 4), dtype=np.uint8)
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

        self.sampler_settings = {}
        wrap = kwargs.pop("wrap", "clamp-to-edge")
        if wrap.startswith("clamp"):
            wrap = "clamp-to-edge"
        self.sampler_settings["address_mode_u"] = wrap
        self.sampler_settings["address_mode_v"] = wrap
        self.sampler_settings["address_mode_w"] = wrap

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
    pass


class ShadertoyChannelCubemapA(ShadertoyChannel):
    pass


# other tabs
class ShadertoyChannelTexture(ShadertoyChannel):
    """
    Represents a shadertoy texture. It is a subclass of `ShadertoyChannel`.
    """

    pass


class ShadertoyChannelCubemap(ShadertoyChannel):
    pass


class ShadertoyChannelVolume(ShadertoyChannel):
    pass


class ShadertoyChannelVideo(ShadertoyChannel):
    pass


class ShadertoyChannelMusic(ShadertoyChannel):
    pass
