# run_example = false
# exploring some issues with this exact shader...
from pathlib import Path

from wgpu_shadertoy import Shadertoy
from wgpu_shadertoy.api import shader_args_from_json

# shadertoy source: https://www.shadertoy.com/view/ssjyWc by FabriceNeyret2 (CC-BY-NC-SA-3.0?)

shader_id = "ssjyWc"
json_path = Path(Path(__file__).parent, f"shader_{shader_id}.json")

shader_args = shader_args_from_json(json_path)
print(f"{shader_args['inputs']=}")
shader = Shadertoy.from_json(json_path, resolution=(1024, 512))
# shader = Shadertoy(**shader_args)

if __name__ == "__main__":
    shader.show()
