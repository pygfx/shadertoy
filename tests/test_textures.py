import numpy as np
from PIL import Image
from pytest import skip
from testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def test_textures_wgsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy, ShadertoyChannelTexture

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

    channel0 = ShadertoyChannelTexture(test_pattern, wrap="repeat", vflip=False)
    channel1 = ShadertoyChannelTexture(gradient)

    shader = Shadertoy(
        shader_code_wgsl, resolution=(640, 480), inputs=[channel0, channel1]
    )
    assert shader.resolution == (640, 480)
    assert shader.shader_code == shader_code_wgsl
    assert shader.shader_type == "wgsl"
    assert shader.inputs[0] == channel0
    assert np.array_equal(shader.inputs[0].data, test_pattern)
    assert shader.inputs[0].sampler_settings["address_mode_u"] == "repeat"
    assert shader.inputs[1] == channel1
    assert np.array_equal(shader.inputs[1].data, gradient)
    assert shader.inputs[1].sampler_settings["address_mode_u"] == "clamp-to-edge"

    shader._draw_frame()


def test_textures_glsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy, ShadertoyChannelTexture

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

    channel0 = ShadertoyChannelTexture(test_pattern, wrap="repeat", vflip="false")
    channel1 = ShadertoyChannelTexture(gradient)

    shader = Shadertoy(shader_code, resolution=(640, 480), inputs=[channel0, channel1])
    assert shader.resolution == (640, 480)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"
    assert shader.inputs[0] == channel0
    assert np.array_equal(shader.inputs[0].data, test_pattern)
    assert shader.inputs[0].sampler_settings["address_mode_u"] == "repeat"
    assert shader.inputs[1] == channel1
    assert np.array_equal(shader.inputs[1].data, gradient)
    assert shader.inputs[1].sampler_settings["address_mode_u"] == "clamp-to-edge"

    shader._draw_frame()


def test_channel_res_wgsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy, ShadertoyChannelTexture

    shader_code_wgsl = """
    fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{
        let uv = frag_coord / i_resolution.xy;
        let c0 = textureSample(i_channel0, sampler0, frag_coord/i_channel_resolution[0].xy);
        let c1 = textureSample(i_channel1, sampler1, frag_coord/i_channel_resolution[1].xy);
        let c2 = textureSample(i_channel2, sampler2, frag_coord/i_channel_resolution[2].xy);
        let c3 = textureSample(i_channel3, sampler3, frag_coord/i_channel_resolution[3].xy);

        let t = vec4<f32>(i_time%8.0);

        // 0 c0, 1 c01, 2 c1, 3 c12, 4 c2, 5 c23, 6 c3, ~7 c30, repeat!
        let c01 = mix(c0, c1, clamp(t-vec4<f32>(1.0), vec4<f32>(0.0), vec4<f32>(1.0)));
        let c23 = mix(c2, c3, clamp(t-vec4<f32>(5.0), vec4<f32>(0.0), vec4<f32>(1.0)));
        let c0123 = mix(c01, c23, clamp(t-vec4<f32>(3.0), vec4<f32>(0.0), vec4<f32>(1.0)));
        
        return c0123;
    }
    """
    img = Image.open("./examples/screenshots/shadertoy_star.png")
    channel0 = ShadertoyChannelTexture(
        img.rotate(0, expand=True), wrap="clamp", vflip=True
    )
    channel1 = ShadertoyChannelTexture(
        img.rotate(90, expand=True), wrap="clamp", vflip=False
    )
    channel2 = ShadertoyChannelTexture(
        img.rotate(180, expand=True), wrap="repeat", vflip=True
    )
    channel3 = ShadertoyChannelTexture(
        img.rotate(270, expand=True), wrap="repeat", vflip=False
    )
    shader = Shadertoy(
        shader_code_wgsl,
        resolution=(1200, 900),
        inputs=[channel0, channel1, channel2, channel3],
    )
    assert shader.resolution == (1200, 900)
    assert shader.shader_code == shader_code_wgsl
    assert shader.shader_type == "wgsl"
    assert len(shader.inputs) == 4
    assert shader._uniform_data["channel_res"] == [
        800.0,
        450.0,
        1.0,
        -99.0,
        450.0,
        800.0,
        1.0,
        -99.0,
        800.0,
        450.0,
        1.0,
        -99.0,
        450.0,
        800.0,
        1.0,
        -99.0,
    ]


def test_channel_res_glsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy, ShadertoyChannelTexture

    shader_code = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec2 uv = fragCoord/iResolution.xy;
        vec4 c0 = texture(iChannel0, fragCoord/iChannelResolution[0].xy);
        vec4 c1 = texture(iChannel1, fragCoord/iChannelResolution[1].xy);
        vec4 c2 = texture(iChannel2, fragCoord/iChannelResolution[2].xy);
        vec4 c3 = texture(iChannel3, fragCoord/iChannelResolution[3].xy);
        
        vec4 t = vec4(mod(iTime,8.0));
        
        // 0 c0, 1 c01, 2 c1, 3 c12, 4 c2, 5 c23, 6 c3, ~7 c30, repeat!
        vec4 c01 = mix(c0, c1, clamp(t-1.0, vec4(0.0), vec4(1.0)));
        vec4 c23 = mix(c2, c3, clamp(t-5.0, vec4(0.0), vec4(1.0)));
        vec4 c0123 = mix(c01, c23, clamp(t-3.0, vec4(0.0), vec4(1.0)));
        
        
        fragColor = c0123;
    }
    """
    img = Image.open("./examples/screenshots/shadertoy_star.png")
    channel0 = ShadertoyChannelTexture(
        img.rotate(0, expand=True), wrap="clamp", vflip=True
    )
    channel1 = ShadertoyChannelTexture(
        img.rotate(90, expand=True), wrap="clamp", vflip=False
    )
    channel2 = ShadertoyChannelTexture(
        img.rotate(180, expand=True), wrap="repeat", vflip=True
    )
    channel3 = ShadertoyChannelTexture(
        img.rotate(270, expand=True), wrap="repeat", vflip=False
    )
    shader = Shadertoy(
        shader_code,
        resolution=(1200, 900),
        inputs=[channel0, channel1, channel2, channel3],
    )
    assert shader.resolution == (1200, 900)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"
    assert len(shader.inputs) == 4
    assert shader._uniform_data["channel_res"] == [
        800.0,
        450.0,
        1.0,
        -99.0,
        450.0,
        800.0,
        1.0,
        -99.0,
        800.0,
        450.0,
        1.0,
        -99.0,
        450.0,
        800.0,
        1.0,
        -99.0,
    ]
