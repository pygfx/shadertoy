import argparse

from .shadertoy import Shadertoy

argument_parser = argparse.ArgumentParser(
    description="Download and render Shadertoy shaders"
)

argument_parser.add_argument(
    "shader_id", type=str, help="The ID of the shader to download and render"
)
argument_parser.add_argument(
    "--resolution",
    type=int,
    nargs=2,
    help="The resolution to render the shader at",
    default=(800, 450),
)

argument_parser.add_argument(
    "--imgui",
    help="automatically turn constants into imgui sliders",
    action="store_true",
)


def main_cli():
    args = argument_parser.parse_args()
    shader_id = args.shader_id
    resolution = args.resolution
    imgui = args.imgui
    # TODO maybe **args?
    shader = Shadertoy.from_id(shader_id, resolution=resolution, imgui=imgui)
    shader.show()


if __name__ == "__main__":
    main_cli()
