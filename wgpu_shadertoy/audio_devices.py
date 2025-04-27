import numpy as np
from abc import ABC, abstractmethod
from collections import deque
import sounddevice as sd
# import logging

# logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
# log = logging.getLogger(__name__)

class AudioDevice(ABC):
    """
    Abstract Base Class for audio input devices used by ShadertoyChannelMusic.
    """

    @abstractmethod
    def get_samples(self, num_samples: int) -> np.ndarray:
        """
        Retrieve the most recent audio samples.

        Args:
            num_samples (int): The number of samples required.

        Returns:
            np.ndarray: A 1D numpy array of float32 audio samples (ideally [-1, 1]),
                        of length `num_samples`. If fewer samples are available than
                        requested, implementations should pad appropriately (e.g., with zeros).
        """
        pass

    @abstractmethod
    def get_rate(self) -> int:
        """
        Returns the sample rate of the audio device in Hz.
        """
        pass

    def start(self):
        """
        Start the audio device (e.g., open microphone stream).
        Optional: Base implementation does nothing.
        """
        pass

    def stop(self):
        """
        Stop the audio device (e.g., close microphone stream).
        Optional: Base implementation does nothing.
        """
        pass

    def is_ready(self) -> bool:
        """
        Check if the device is ready or has sufficient data.
        Optional: Base implementation assumes always ready.
        """
        return True
    
    def gain(self) -> float:
        """
        Returns the gain factor for the audio device.
        Optional: Base implementation returns 1.0 (no gain).
        """
        return 1.0
    
class FIFOPushAudioDevice(AudioDevice):
    """
    An AudioDevice implementation using an internal FIFO buffer (deque).
    Samples are added using the `push_samples` method.
    """
    def __init__(self, rate: int, max_buffer_samples: int, gain: float = 0.6):
        self._gain = gain
        self._rate = rate
        # Buffer stores slightly more than needed for FFT to handle requests
        self._buffer = deque(maxlen=max_buffer_samples)

    def get_rate(self) -> int:
        return self._rate

    def push_samples(self, new_samples: np.ndarray):
        """Appends new audio samples to the internal buffer."""
        # Ensure input is float32? Or handle conversion? Assume float for now.
        self._buffer.extend(new_samples.astype(np.float32))

    def get_samples(self, num_samples: int) -> np.ndarray:
        """Returns the most recent `num_samples` from the buffer."""
        current_buffer = np.array(self._buffer) # Convert deque to numpy array for slicing
        available_samples = current_buffer.size

        if available_samples >= num_samples:
            # Return the last num_samples
            return current_buffer[-num_samples:]
        else:
            # Not enough samples, return what we have padded with leading zeros
            padded_samples = np.zeros(num_samples, dtype=np.float32)
            if available_samples > 0:
                padded_samples[-available_samples:] = current_buffer
            return padded_samples

    def is_ready(self) -> bool:
        # Consider ready if buffer has at least enough samples for typical request?
        # Let's say ready if buffer has at least 512 samples (typical fft_input_size)
        return len(self._buffer) >= 512
    
    def gain(self) -> float:
        return self._gain
    
class NullAudioDevice(AudioDevice):
    """An AudioDevice that always returns silence."""
    def __init__(self, rate: int = 44100):
        self._rate = rate

    def get_rate(self) -> int:
        return self._rate

    def get_samples(self, num_samples: int) -> np.ndarray:
        return np.zeros(num_samples, dtype=np.float32)
    
class NoiseAudioDevice(AudioDevice):
    """An AudioDevice that always returns silence."""
    def __init__(self, rate: int = 44100):
        self._rate = rate

    def get_rate(self) -> int:
        return self._rate

    def get_samples(self, num_samples: int) -> np.ndarray:
        return np.random.uniform(-1, 1, num_samples).astype(np.float32)
    
class MicrophoneAudioDevice(FIFOPushAudioDevice):
    """
    An AudioDevice implementation that reads live audio from a system input device
    using the `sounddevice` library. Provides single-channel (mono) float32 samples.
    """
    def __init__(self,
                 sample_rate: int = 44100,
                 buffer_duration_seconds: float = 5.0,
                 chunk_size: int = 1024,
                 device_index: int | None = None,
                 gain: float = 0.6):
        """
        Initializes the MicrophoneAudioDevice.

        Args:
            sample_rate (int): The desired sample rate in Hz.
            buffer_duration_seconds (float): The duration of the internal FIFO buffer
                                             in seconds. Determines the maximum amount
                                             of recent audio history stored.
            chunk_size (int): The number of samples to read from the audio device
                              in each callback chunk. Affects latency and overhead.
            device_index (int | None): The index of the input audio device to use.
                                       If None, the default system input device is used.
                                       Use `sounddevice.query_devices()` to list devices.
            gain (float): The gain factor to apply to the audio fft data.
        """
        max_samples = int(sample_rate * buffer_duration_seconds)
        super().__init__(rate=sample_rate, max_buffer_samples=max_samples, gain=gain)

        self._chunk_size = chunk_size
        self._device_index = device_index
        self._stream = None
        # print(f"MicrophoneAudioDevice initialized. Rate: {sample_rate} Hz, "
        #          f"Buffer: {buffer_duration_seconds}s ({max_samples} samples), "
        #          f"Chunk Size: {chunk_size}, Device: {device_index or 'Default'}")

    def _audio_callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """
        This function is called by sounddevice in a separate thread
        whenever new audio data is available.
        """
        if status:
            print(f"Warning: Sounddevice callback status: {status}")
            if status.input_underflow: print(f"Warning: Input underflow detected!")
            if status.input_overflow: print(f"Warning: Input overflow detected! Buffer might be too small or processing too slow.")

        # indata comes in as float32 (due to dtype='float32' in stream)
        # It might be multi-channel, but we requested channels=1,
        # so it should have shape (frames, 1). We need 1D.
        if indata.shape[1] != 1:
             # This shouldn't happen if channels=1 works, but as a fallback:
             mono_samples = indata.mean(axis=1).astype(np.float32)
        else:
             mono_samples = indata[:, 0] # Extract the single channel, makes it 1D

        # Push the mono samples into the deque buffer (managed by parent class)
        # This uses self._buffer.extend(), which is thread-safe in CPython.
        self.push_samples(mono_samples)

    def start(self):
        """Starts the audio stream from the microphone."""
        if self._stream is not None and self._stream.active:
            print("Warning: Stream already running. Call stop() first if you want to restart.")
            return

        try:
            self._stream = sd.InputStream(
                samplerate=self.get_rate(),
                blocksize=self._chunk_size,
                device=self._device_index,
                channels=1,  # Request mono directly
                dtype=np.float32, # Request 32-bit float
                callback=self._audio_callback,
                latency='low' # Or 'high' for potentially more stable streaming
            )
            self._stream.start()
        except Exception as e:
            print(f"Error: Failed to start audio stream: {e}")
            self._stream = None # Ensure stream is None if start failed
            # Re-raise the exception if the calling code needs to know
            raise e

    def stop(self):
        """Stops the audio stream."""
        if self._stream is None:
            return

        try:
            if self._stream.active:
                self._stream.stop()
            self._stream.close()
        except Exception as e:
            print(f"Error: stopping audio stream: {e}")
        finally:
            # Ensure stream is marked as stopped regardless of errors
            self._stream = None

    # Optional: Override is_ready to also check the stream status
    def is_ready(self) -> bool:
        """Check if the stream is active and the buffer has sufficient data."""
        parent_ready = super().is_ready()
        stream_active = self._stream is not None and self._stream.active
        # TODO - allow getting samples
        # even if stream just stopped but buffer is full.
        return stream_active and parent_ready