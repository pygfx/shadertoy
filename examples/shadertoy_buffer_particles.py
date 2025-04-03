from pathlib import Path

from wgpu_shadertoy import Shadertoy

# shadertoy source: https://www.shadertoy.com/view/3tBGzh by wyatt (CC-BY-NC-SA-3.0?)

json_path = Path(Path(__file__).parent, "shader_3tBGzh.json")
shader = Shadertoy.from_json(json_path)

if __name__ == "__main__":
    shader.show()
