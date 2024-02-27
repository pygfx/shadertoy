# run_example = false

from wgpu_shadertoy import Shadertoy

# shadertoy source: https://www.shadertoy.com/view/MllSzX by demofox CC-BY-NC-SA-3.0
shader = Shadertoy.from_json(".\examples\shader_MllSzX.json")

if __name__ == "__main__":
    shader.show()
