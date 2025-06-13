import argparse
from wgpu_shadertoy import Shadertoy
from wgpu_shadertoy import MicrophoneAudioDevice
from pathlib import Path
import sounddevice as sd
import sys

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='Run a Shadertoy shader with an audio input device.')
parser.add_argument('--from_id', type=str, default="llSGDh", 
                    help='Shadertoy ID (default: llSGDh) https://www.shadertoy.com/view/llSGDh by iq CC-BY-NC-SA-3.0')
parser.add_argument('--list-audio-devices', action='store_true',
                    help='List all available audio input devices and exit')
parser.add_argument('--device-index', type=int, default=None,
                    help='Audio device index to use (default: system default)')

json_path = Path(Path(__file__).parent, "shader_llSGDh.json")

if __name__ == "__main__":
    # Parse the command line arguments
    args = parser.parse_args()

    # If list-audio-devices flag is present, list devices and exit
    if args.list_audio_devices:
        print("Available audio input devices:")
        print(sd.query_devices())
        sys.exit(0)

    # Use the provided ID or the default one
    shader_id = args.from_id

    # Use the device index from command line if provided
    # if device_index is None, sounddevice will use the system default device
    device_index = args.device_index

    # Create microphone audio device with specified device index (or None for default)
    #   We could use a NoiseAudioDevice or NullAudioDevice for testing without a mic: audio_device = NoiseAudioDevice(rate=44100)
    audio_device = MicrophoneAudioDevice(device_index=device_index, sample_rate=44100, buffer_duration_seconds=2.0)
    audio_device.start()

    # shadertoy source: https://www.shadertoy.com/view/llSGDh by iq CC-BY-NC-SA-3.0
    shader = None
    if shader_id == "llSGDh":
        # Load the shader from JSON file
        shader = Shadertoy.from_json(json_path, use_cache=True, audio_device=audio_device)
    else:
        # Load the shader from Shadertoy by ID
        shader = Shadertoy.from_id(shader_id, use_cache=True, audio_device=audio_device)

    shader.show()