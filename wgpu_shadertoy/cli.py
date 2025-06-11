import argparse
import sounddevice as sd
import sys

from .shadertoy import Shadertoy
from .audio_devices import MicrophoneAudioDevice

argument_parser = argparse.ArgumentParser(
    description="Download and render Shadertoy shaders"
)

argument_parser.add_argument(
    "shader_id", 
    type=str, 
    nargs='?',  # Make it optional in case of listing audio devices
    help="The ID of the shader to download and render"
)
argument_parser.add_argument(
    "--resolution",
    type=int,
    nargs=2,
    help="The resolution to render the shader at",
    default=(800, 450),
)
argument_parser.add_argument('--list-audio-devices', 
    action='store_true',
    help='List all available audio input/output devices and exit',
)
argument_parser.add_argument('--enable-audio-input',
    action='store_true',
    help='Enable audio input (default: False)',
)
argument_parser.add_argument('--audio-input-index',
    type=int,
    default=None,
    help='Audio device index to use for input (default: system default)',
)

def main_cli():
    args = argument_parser.parse_args()

    # If list-audio-devices flag is present, list devices and exit
    if args.list_audio_devices:
        print("Available audio input devices:")
        print(sd.query_devices())
        sys.exit(0)

    # Check if shader_id is provided when not listing audio devices
    if args.shader_id is None:
        argument_parser.error("the following arguments are required: shader_id")

    shader_id = args.shader_id
    resolution = args.resolution
    shader = None

    if args.enable_audio_input:
        # Create audio input device with specified device index (or None for default)
        audio_device = MicrophoneAudioDevice(device_index=args.audio_input_index, sample_rate=44100, buffer_duration_seconds=2.0)
        audio_device.start()
        shader = Shadertoy.from_id(shader_id, resolution=resolution, audio_device=audio_device)
    else:
        shader = Shadertoy.from_id(shader_id, resolution=resolution)

    shader.show()


if __name__ == "__main__":
    main_cli()
