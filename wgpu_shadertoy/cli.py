import argparse
import os

from .shadertoy import Shadertoy
from .record import record_offscreen

argument_parser = argparse.ArgumentParser(
    description="Download and render Shadertoy shaders"
)

argument_parser.add_argument(
    "shader_id", type=str, help="The ID of the shader to download and render"
)
# shared args
argument_parser.add_argument(
    "--resolution",
    type=int,
    nargs=2,
    help="The resolution to render the shader at",
    default=(800, 450),
)
# maybe put framerate here

command_parser = argument_parser.add_subparsers(dest="command", help="subcommands for this CLI")

show_parser = command_parser.add_parser(
    "show",
    help="display the shader in a GUI (default)"
)
# TODO vsync, max framerate(?), gui lib, maybe offsets?

record_parser = command_parser.add_parser(
    "record",
    help="records shader to a video file (offscreen)"
)
record_parser.add_argument(
    "--output_file",
    type=str,
    default=None, #maybe the shader id or name or something?
    help="The output file to save the recorded video"
)
record_parser.add_argument(
    "--start_offset",
    type=float,
    default=0.0,
    help="the starting iTime, not prerendering frames", # maybe worth it for accumulation?
)
record_parser.add_argument(
    "--duration",
    type=float,
    default=10.0,
    help="The duration of the recorded video in seconds, defaults to 10.0",
)
record_parser.add_argument(
    "--framerate",
    type=int,
    default=60,
    help="The framerate of the recorded video, defaults to 60",
)
record_parser.add_argument(
    "--target_size",
    type=float,
    default=9.9,
    help="The target size of the recorded video, defaults to 9.9 MB",
)
# maybe bitrate too?

def main_cli():
    args = argument_parser.parse_args()
    shader_id = args.shader_id.rstrip('/').split('/')[-1]
    resolution = args.resolution
    if args.command == "record":
        recording_args = {
            "start_offset": args.start_offset,
            "duration": args.duration,
            "framerate": args.framerate,
            "target_size": args.target_size,
            "output_file": args.output_file or f"{shader_id}.mp4",
        }
        # TODO: replace resolution with padded variant here?
        shader = Shadertoy.from_id(shader_id, resolution=resolution, offscreen=True)
        record_offscreen(shader, **recording_args)
        print(f"Recording finished: {os.getcwd()}/{recording_args['output_file']}")
    else:
        # gui-args = ?
        shader = Shadertoy.from_id(shader_id, resolution=resolution)
        shader.show()


if __name__ == "__main__":
    main_cli()
