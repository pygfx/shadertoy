from pathlib import Path

from wgpu_shadertoy import Shadertoy

# shadertoy source: https://www.shadertoy.com/view/ssjyWc by FabriceNeyret2 (CC-BY-NC-SA-3.0?)

# current "bug": the string kinda floats off to the upper right corner, without any inputs... ? Likely to be some issue with the implementation of buffers.
shader_id = "ssjyWc"
json_path = Path(Path(__file__).parent, f"shader_{shader_id}.json")

shader = Shadertoy.from_json(json_path, resolution=(1024, 512))

if __name__ == "__main__":
    shader.show()
