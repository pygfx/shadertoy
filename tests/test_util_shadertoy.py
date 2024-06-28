from pytest import skip
from testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def test_shadertoy_wgsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
            let uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }else{
                return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "wgsl"

    shader._draw_frame()


def test_shadertoy_wgsl2():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        fn shader_main(frag_coord: vec2<f32>) -> vec4<f32> {
            let uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }else{
                return vec4<f32>( 0.5 + 0.5 * sin(i_time * vec3<f32>(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, shader_type="wgsl", resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "wgsl"

    shader._draw_frame()


def test_shadertoy_glsl():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        void shader_main(out vec4 fragColor, vec2 frag_coord) {
            vec2 uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(i_time * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"

    shader._draw_frame()


def test_shadertoy_glsl2():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
            vec2 uv = fragCoord / iResolution.xy;

            if ( length(fragCoord - iMouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(iTime * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, shader_type="glsl", resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"

    shader._draw_frame()


def test_shadertoy_glsl3():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        void    mainImage( out vec4 fragColor, in vec2 fragCoord ) {
            vec2 uv = fragCoord / iResolution.xy;

            if ( length(fragCoord - iMouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(iTime * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"

    shader._draw_frame()


def test_shadertoy_offscreen():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        void shader_main(out vec4 fragColor, vec2 frag_coord) {
            vec2 uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(i_time * vec3(uv, 1.0) ), 1.0);
            }

        }
    """
    # kinda redundant, tests are run with force_offscreen anyway
    shader = Shadertoy(shader_code, resolution=(800, 450), offscreen=True)
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"
    assert shader._offscreen is True


def test_shadertoy_snapshot():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        void shader_main(out vec4 fragColor, vec2 frag_coord) {
            vec2 uv = frag_coord / i_resolution.xy;

            if ( length(frag_coord - i_mouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(i_time * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    shader = Shadertoy(shader_code, resolution=(800, 450), offscreen=True)
    frame1a = shader.snapshot(
        time_float=0.0,
        mouse_pos=(
            0,
            0,
            0,
            0,
        ),
    )
    frame2a = shader.snapshot(
        time_float=1.2,
        mouse_pos=(
            100,
            200,
            0,
            0,
        ),
    )
    frame1b = shader.snapshot(
        time_float=0.0,
        mouse_pos=(
            0,
            0,
            0,
            0,
        ),
    )
    frame2b = shader.snapshot(
        time_float=1.2,
        mouse_pos=(
            100,
            200,
            0,
            0,
        ),
    )

    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"
    assert shader._offscreen is True
    assert frame1a == frame1b
    assert frame2a == frame2b


def test_shadertoy_with_buffers():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import BufferRenderPass, Shadertoy, ShadertoyChannelBuffer

    image_code = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec2 uv = fragCoord/iResolution.xy;
        
        vec4 c0 = texture(iChannel0, uv);
        vec4 c1 = texture(iChannel1, uv);

        fragColor = vec4(c0.r, c0.g, c1.b, c1.a);
    }
    """

    buffer_code_a = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        fragColor = vec4(fragCoord.x/iResolution.x);
    }
    """
    buffer_code_b = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        fragColor = vec4(fragCoord.y/iResolution.y);
    }
    """

    buffer_pass_a = BufferRenderPass(buffer_idx="a", code=buffer_code_a)
    buffer_pass_b = BufferRenderPass(buffer_idx="b", code=buffer_code_b)
    channel_a = ShadertoyChannelBuffer(buffer=buffer_pass_a)
    channel_b = ShadertoyChannelBuffer(buffer="b", wrap="repeat")
    shader = Shadertoy(
        shader_code=image_code,
        resolution=(800, 450),
        inputs=[channel_a, channel_b],
        buffers={"a": buffer_pass_a, "b": buffer_pass_b},
    )

    assert shader.resolution == (800, 450)
    assert shader.buffers["a"].shader_code == buffer_code_a
    assert shader.buffers["b"].shader_code == buffer_code_b
    assert shader.image.channels[0].renderpass.buffer_idx == "a"
    assert shader.image.channels[1].renderpass.buffer_idx == "b"
    assert shader.image.channels[1].sampler_settings["address_mode_u"] == "repeat"


def test_shadertoy_with_buffer_missing():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import (
        BufferRenderPass,
        Shadertoy,
        ShadertoyChannelBuffer,
        ShadertoyChannelTexture,
    )

    image_code = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        vec2 uv = fragCoord/iResolution.xy;
        
        vec4 c0 = texture(iChannel0, uv);
        vec4 c1 = texture(iChannel1, uv);

        fragColor = vec4(c0.r, c0.g, c1.b, c1.a);
    }
    """

    buffer_code = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        fragColor = vec4(fragCoord.x/iResolution.x);
    }
    """

    buffer_pass_a = BufferRenderPass(buffer_idx="a", code=buffer_code)
    channel_a = ShadertoyChannelBuffer(buffer=buffer_pass_a)
    # this references the buffer "b" we don't have attched. We use the default 8x8 pixels of black instead.
    channel_b = ShadertoyChannelBuffer(buffer="b", wrap="repeat")
    shader = Shadertoy(
        shader_code=image_code,
        resolution=(800, 450),
        inputs=[channel_a, channel_b],
        buffers={"a": buffer_pass_a},
    )

    assert shader.resolution == (800, 450)
    assert shader.buffers["a"].shader_code == buffer_code
    assert shader.image.channels[0].renderpass.buffer_idx == "a"
    assert type(shader.image.channels[1]) == ShadertoyChannelTexture
    assert not shader.image.channels[1].data[0:2].any()
