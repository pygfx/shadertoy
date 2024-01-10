from pytest import skip
from testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def test_textures_wgsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy, ShadertoyChannel

    shader_code_wgsl = """
    fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{
        let uv = frag_coord / i_resolution.xy;
        let c0 = textureSample(i_channel0, sampler0, 2.0*uv);
        let c1 = textureSample(i_channel1, sampler1, 3.0*uv);
        return mix(c0,c1,abs(sin(i_time)));
    }
    """
    test_pattern = memoryview(
        bytearray((int(i != k) * 255 for i in range(8) for k in range(8))) * 4
    ).cast("B", shape=[8, 8, 4])
    gradient = memoryview(
        bytearray((i for i in range(0, 255, 8) for _ in range(4))) * 32
    ).cast("B", shape=[32, 32, 4])

    channel0 = ShadertoyChannel(test_pattern, wrap="repeat")
    channel1 = ShadertoyChannel(gradient)

    shader = Shadertoy(
        shader_code_wgsl, resolution=(640, 480), inputs=[channel0, channel1]
    )
    assert shader.resolution == (640, 480)
    assert shader.shader_code == shader_code_wgsl
    assert shader.shader_type == "wgsl"
    assert shader.inputs[0] == channel0
    assert shader.inputs[0].data == test_pattern
    assert shader.inputs[0].sampler_settings["address_mode_u"] == "repeat"
    assert shader.inputs[1] == channel1
    assert shader.inputs[1].data == gradient
    assert shader.inputs[1].sampler_settings["address_mode_u"] == "clamp-to-edge"

    shader._draw_frame()


def test_textures_glsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy, ShadertoyChannel

    shader_code = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec2 uv = fragCoord/iResolution.xy;
        vec4 c0 = texture(iChannel0, 2.0*uv);
        vec4 c1 = texture(iChannel1, 3.0*uv);
        fragColor = mix(c0,c1,abs(sin(i_time)));
    }
    """

    test_pattern = memoryview(
        bytearray((int(i != k) * 255 for i in range(8) for k in range(8))) * 4
    ).cast("B", shape=[8, 8, 4])
    gradient = memoryview(
        bytearray((i for i in range(0, 255, 8) for _ in range(4))) * 32
    ).cast("B", shape=[32, 32, 4])

    channel0 = ShadertoyChannel(test_pattern, wrap="repeat")
    channel1 = ShadertoyChannel(gradient)

    shader = Shadertoy(shader_code, resolution=(640, 480), inputs=[channel0, channel1])
    assert shader.resolution == (640, 480)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"
    assert shader.inputs[0] == channel0
    assert shader.inputs[0].data == test_pattern
    assert shader.inputs[0].sampler_settings["address_mode_u"] == "repeat"
    assert shader.inputs[1] == channel1
    assert shader.inputs[1].data == gradient
    assert shader.inputs[1].sampler_settings["address_mode_u"] == "clamp-to-edge"

    shader._draw_frame()
