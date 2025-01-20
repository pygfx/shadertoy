from typing import Tuple

import numpy as np
import wgpu


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
        if channel_idx is not None:
            channel_idx = kwargs.pop("channel_idx", None)
        self._channel_idx:int = channel_idx
        self.args = args
        self.kwargs = kwargs
        self.dynamic:bool = False

        @property
        def sampler_settings(sef) -> dict:
            """
            Sampler settings for this channel. Wrap currently supported. Filter not yet.
            """
            settings = {}
            wrap = kwargs.pop("wrap", "clamp-to-edge")
            # "warp", "clamp" or "repeat" is what we should expect on Shadertoy
            if wrap.startswith("clamp"):
                wrap = "clamp-to-edge"
            settings["address_mode_u"] = wrap
            settings["address_mode_v"] = wrap
            # we don't do 3D textures yet, but I guess ssetting this too is fine.
            settings["address_mode_w"] = wrap
            return settings

        @property
        def parent(self) -> "RenderPass":
            """
            Parent renderpass of this channel.
            """
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
        def channel_res(self) -> Tuple[int, int, int, int]:
            """
            Tuple of (width, height, pixel_aspect=1, padding=-99)
            """
            return (self.size[1], self.size[0], 1, -99)
        


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
                return ShadertoyChannelTexture(*args, channel_idx=self._channel_idx, **kwargs)
            else:
                raise NotImplementedError(f"Doesn't support {self.ctype=} yet")

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
        #destination, data, data_layout, size
        device.queue.write_texture(
            destination={
                "texture": texture,
            },
            data=self.data,
            data_layout={
                "bytes_per_row": self.data.strides[0], #multiple of 256
                "rows_per_image": self.data.size[0], 
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
