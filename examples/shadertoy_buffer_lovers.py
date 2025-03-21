from wgpu_shadertoy import Shadertoy

# shadertoy source: https://www.shadertoy.com/view/ssjyWc by FabriceNeyret2 (CC-BY-NC-SA-3.0?)

# TODO: replace with json so it can run without a API key?
shader = Shadertoy.from_id("https://www.shadertoy.com/view/ssjyWc")

if __name__ == "__main__":
    shader.show()