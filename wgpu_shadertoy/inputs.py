from typing import Tuple

import numpy as np
from numpy.fft import rfft
import wgpu
from .audio_devices import AudioDevice, NullAudioDevice

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
        self.dynamic: bool = kwargs.pop("dynamic", False)

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
        return settings

    @property
    def parent(self):
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
        elif self.ctype == "mic":
            return ShadertoyChannelMusic(*args, **kwargs)
        else:
            raise NotImplementedError(f"Doesn't support {self.ctype=} yet")

    def _update_input(self, time, time_delta):
        """
        Updates the input channel. This is called by the parent renderpass.
        """
        pass

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
    """
    Represents a Shadertoy music input channel (iChannel for audio).
    Uses an AudioDevice to fetch samples and updates a 2D texture
    (typically 512x2, rg32float) with FFT and Waveform data.

    Data Layout:
    - The internal NumPy array has shape (height, width, components),
      e.g., (2, 512, 2) for float32 dtype.
    - Component 0 (R channel), Row 0 (y~0.25): Contains FFT data [0, 1].
    - Component 0 (R channel), Row 1 (y~0.75): Contains Waveform data (often normalized to [0, 1]).
    - Component 1 (G channel): Currently unused (contains zeros).
    - In the shader (GLSL):
        - Access FFT via `texture(iChannel0, vec2(u, 0.25)).r` (or .x)
        - Access Wave via `texture(iChannel0, vec2(u, 0.75)).r` (or .x)
      (Texture coordinates assume y=0 is bottom, y=1 is top;
       0.25 targets the middle of the first row, 0.75 targets the middle of the second row).

    Parameters:
        audio_device (AudioDevice): An instance of an AudioDevice subclass
                                    providing the audio samples.
        size (tuple): Texture size as (width, height), e.g., (512, 2).
                      Width determines the texture resolution for wave/FFT.
        dynamic (bool): If True, allows the texture to be updated later. Default is True.
        **kwargs: Additional arguments passed to the base ShadertoyChannel.
    """

    def __init__(self, size=(512, 2), dynamic=True, **kwargs):
            kwargs['ctype'] = 'music'
            # Ensure dynamic is True if we have an audio device providing updates
            super().__init__(dynamic=dynamic, **kwargs)

            # check if kwargs has 'audio_device' and set it
            audio_device = kwargs.pop("audio_device", None)

            if audio_device is None:
                # use the NULL audio device
                audio_device = NullAudioDevice(44100)

            self.audio_device = audio_device
            self.format = wgpu.TextureFormat.rg32float
            self._num_components = 2
            self._gain = self.audio_device.gain()

            self.texture_width, self.texture_height = size
            if self.texture_height != 2:
                print(f"Warning: ShadertoyChannelMusic typically uses height=2 (FFT, Wave), got {self.texture_height}")

            # Initialize placeholder data matching texture structure
            self.data = np.zeros((self.texture_height, self.texture_width, self._num_components), dtype=np.float32)
            self.texture_size = (self.texture_width, self.texture_height, 1)

            self._texture = None
            self._device = None

    @property
    def channel_res(self) -> Tuple[int, int, int, int]:
        """
        Returns the resolution of the channel texture: (width, height, depth=1, padding=-99)
        """
        return (self.texture_size[0], self.texture_size[1], 1, -99)

    @property
    def size(self) -> Tuple:
        """ Size of the data array (height, width, components) """
        return self.data.shape

    def bind_texture(self, device: wgpu.GPUDevice) -> Tuple[list, list]:
        """
        Prepares the texture and sampler for the audio data.
        Returns its binding layouts and bindgroup layout entries.
        """
        self._device = device # Store device reference for updates

        binding_layout = self._binding_layout()

        usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST

        if self.dynamic:
            usage |= wgpu.TextureUsage.COPY_DST

        # Create the texture only if it doesn't exist
        if self._texture is None:
            self._texture = device.create_texture(
                size=self.texture_size,
                format=self.format,
                usage=usage,
                label="shadertoy-music-channel"
            )

            # Perform initial write
            bytes_per_row = self.texture_size[0] * self._num_components * self.data.itemsize
            if bytes_per_row % 256 != 0:
                 bytes_per_row += 256 - (bytes_per_row % 256)

            device.queue.write_texture(
                destination={"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
                data=self.data,
                data_layout={
                    "bytes_per_row": bytes_per_row,
                    "rows_per_image": self.texture_size[1], # height
                },
                size=self.texture_size, # (width, height, depth)
            )

        texture_view = self._texture.create_view()
        sampler = device.create_sampler(**self.sampler_settings)

        bind_groups_layout_entry = self._bind_groups_layout_entries(
            texture_view, sampler
        )

        return binding_layout, bind_groups_layout_entry

    def update(self, new_data: np.ndarray):
        """
        Updates the texture data with a pre-formatted NumPy array.
        The array must have the shape (height, width, 2) and dtype (float32).

        Args:
            new_data: New audio data (e.g., shape (2, 512, 2), dtype float32).
        """
        if not self.dynamic:
            raise TypeError("This ShadertoyChannelMusic was not initialized with dynamic=True")
        if self._texture is None or self._device is None:
            raise RuntimeError("Texture hasn't been bound to a device yet via bind_texture()")

        new_data = np.asarray(new_data, dtype=np.float32)
        if new_data.shape != self.data.shape:
             raise ValueError(f"New data shape {new_data.shape} must match original shape {self.data.shape}")

        new_data_contiguous = np.ascontiguousarray(new_data)
        self.data = new_data # Update internal reference if needed outside

        bytes_per_row = self.texture_size[0] * self._num_components * new_data_contiguous.itemsize
        if bytes_per_row % 256 != 0:
            bytes_per_row += 256 - (bytes_per_row % 256)

        self._device.queue.write_texture(
            destination={"texture": self._texture, "mip_level": 0, "origin": (0, 0, 0)},
            data=new_data_contiguous,
            data_layout={
                "bytes_per_row": bytes_per_row,
                "rows_per_image": self.texture_size[1], # height
            },
            size=self.texture_size, # (width, height, depth)
        )

    def update_from_audio(self, rate: int, samples: np.ndarray):
        """
        Processes raw audio samples to generate FFT and Waveform data,
        then updates the internal texture. Both FFT and Waveform are
        packed into the first component (R channel) of their respective rows.

        Args:
            rate (int): The sample rate of the audio data. (Currently unused in processing)
            samples (np.ndarray): A 1D NumPy array of float32 audio samples.
                                  Should have length == fft_input_size.
        """
        # Keep the checks here as this method might be called externally too
        if not self.dynamic:
            # print("Warning: Calling update_from_audio on non-dynamic channel.") # Debug
            return
        if self._texture is None or self._device is None:
            print("Warning: update_from_audio called before texture is bound.")
            return

        target_width = self.texture_width
        fft_input_size = target_width * 2

        # --- Assume 'samples' already has the correct length (fft_input_size) ---
        # The check for length should happen in the caller (_update_input)

        # --- Process Waveform Data ---
        # Use the *last* target_width samples from the input buffer
        wave_segment = samples[-target_width:]
        # Normalize waveform to [0, 1] range for .x access in shader
        wave_segment_normalized = (np.clip(wave_segment, -1.0, 1.0) + 1.0) * 0.5

        # --- Process FFT Data ---
        fft_segment = samples
        window = np.hanning(fft_input_size)
        fft_segment_windowed = fft_segment * window
        fft_result = rfft(fft_segment_windowed)
        fft_magnitude = np.abs(fft_result[1:target_width+1]) # Skip DC

        # Apply square root to magnitude to boost lower values
        fft_sqrt_magnitude = np.sqrt(fft_magnitude)

        # Normalize/Scale: Adjust this scaling factor based on visual results
        # This factor depends heavily on the input audio level and desired visual range.
        fft_processed = fft_sqrt_magnitude * self._gain

        # Clip to ensure the final value is in the [0, 1] range for the texture
        fft_final = np.clip(fft_processed, 0.0, 1.0)

        # --- Format for Texture ---
        texture_data = np.zeros((self.texture_height, self.texture_width, self._num_components), dtype=np.float32)

        if self.texture_height == 2:
            # Store FFT in the first row (index 0), R component (index 0)
            texture_data[0, :, 0] = fft_final
            # Store Waveform in the second row (index 1), R component (index 0)
            texture_data[1, :, 0] = wave_segment_normalized
        else:
             print(f"Warning: update_from_audio packing only implemented for height=2")
             if self.texture_height == 1:
                  texture_data[0, :, 0] = fft_final
                  # Decide how to pack wave for height=1, maybe G?
                  # texture_data[0, :, 1] = wave_segment_normalized

        # --- Update GPU Texture ---
        # Call the existing self.update method which handles the wgpu call
        self.update(texture_data)


    # --- _update_input now fetches data and calls update_from_audio ---
    def _update_input(self, time: float, time_delta: float):
        """
        Fetches the latest audio samples from the audio device and triggers
        the processing and texture update via update_from_audio.
        Called by the parent renderpass before drawing.

        Args:
            time (float): Current shader time (iTime).
            time_delta (float): Time since last frame (iTimeDelta).
        """
        # Keep initial checks
        if not self.dynamic:
            return
        if not self.audio_device.is_ready():
            return
        if self._texture is None or self._device is None:
            print("Warning: _update_input called before texture is bound.")
            return

        # Determine how many samples are needed for processing
        target_width = self.texture_width
        fft_input_size = target_width * 2
        rate = self.audio_device.get_rate()

        # Fetch the required number of samples from the device
        samples = self.audio_device.get_samples(fft_input_size)

        # Ensure we got the right amount (device might pad if not enough)
        # This check ensures update_from_audio receives the expected size
        if samples.size != fft_input_size:
             print(f"Warning: Audio device returned {samples.size} samples, expected {fft_input_size}. Padding.")
             temp_samples = np.zeros(fft_input_size, dtype=np.float32)
             valid_samples = min(samples.size, fft_input_size)
             temp_samples[-valid_samples:] = samples[-valid_samples:]
             samples = temp_samples

        # --- Delegate processing and GPU update to update_from_audio ---
        self.update_from_audio(rate, samples)

    def _binding_layout(self):
        return [
            {
                "binding": self.texture_binding,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.unfilterable_float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                    "multisampled": False,
                },
            },
            {
                "binding": self.sampler_binding,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "sampler": {"type": wgpu.SamplerBindingType.non_filtering},
            },
        ]

    @property
    def sampler_settings(self) -> dict:
        settings = super().sampler_settings
        # we're using Rg32Float so we probably won't be able to filter:
        # settings["mag_filter"] = wgpu.FilterMode.linear
        # settings["min_filter"] = wgpu.FilterMode.linear
        settings["address_mode_u"] = wgpu.AddressMode.clamp_to_edge
        settings["address_mode_v"] = wgpu.AddressMode.clamp_to_edge
        settings["address_mode_w"] = wgpu.AddressMode.clamp_to_edge
        return settings