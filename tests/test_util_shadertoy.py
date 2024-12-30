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
        void mainImage(out vec4 fragColor, vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;

            if ( length(fragCoord - iMouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(iTime * vec3(uv, 1.0) ), 1.0);
            }

        }
    """

    # tests the shader_type detection base case we will most likely see.
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

    # this tests if setting the shader_type to glsl works as expected
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

    # this tests glsl detection against the regular expression when using more than one whitespace between void and mainImage.
    shader = Shadertoy(shader_code, resolution=(800, 450))
    assert shader.resolution == (800, 450)
    assert shader.shader_code == shader_code
    assert shader.shader_type == "glsl"

    shader._draw_frame()


def test_shadertoy_offscreen():
    # Import here, because it imports the wgpu.gui.auto
    from wgpu_shadertoy import Shadertoy

    shader_code = """
        void mainImage(out vec4 fragColor, vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;

            if ( length(fragCoord - iMouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(iTime * vec3(uv, 1.0) ), 1.0);
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
        void mainImage(out vec4 fragColor, vec2 fragCoord) {
            vec2 uv = fragCoord / iResolution.xy;

            if ( length(fragCoord - iMouse.xy) < 20.0 ) {
                fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            }else{
                fragColor = vec4( 0.5 + 0.5 * sin(iTime * vec3(uv, 1.0) ), 1.0);
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
        vec4 c2 = texture(iChannel2, uv);
        vec4 c3 = texture(iChannel3, uv);

        fragColor = vec4(c0.r, c1.g, c2.b, c3.a);
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
    buffer_code_c = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        fragColor = vec4(fragCoord.x/iResolution.y);
    }
    """
    buffer_code_d = """
    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        fragColor = vec4(fragCoord.y/iResolution.x);
    }
    """

    # this tests a combination of creating explicit and implicit BufferRenderPass classes for the always explicit ChannelBuffer instances.
    buffer_pass_a = BufferRenderPass(buffer_idx="a", code=buffer_code_a)
    buffer_pass_b = BufferRenderPass(buffer_idx="b", code=buffer_code_b)
    channel_0 = ShadertoyChannelBuffer(buffer=buffer_pass_a)
    channel_1 = ShadertoyChannelBuffer(buffer="b", wrap="repeat")
    channel_2 = ShadertoyChannelBuffer(buffer="c")
    channel_3 = ShadertoyChannelBuffer(buffer="d", wrap="clamp")

    shader = Shadertoy(
        shader_code=image_code,
        resolution=(800, 450),
        inputs=[channel_0, channel_1, channel_2, channel_3],
        buffers={
            "a": buffer_pass_a,
            "b": buffer_pass_b,
            "c": buffer_code_c,
            "d": buffer_code_d,
        },
    )

    assert shader.resolution == (800, 450)
    assert shader.buffers["a"].shader_code == buffer_code_a
    assert shader.buffers["b"].shader_code == buffer_code_b
    assert shader.buffers["c"].shader_code == buffer_code_c
    assert shader.buffers["d"].shader_code == buffer_code_d
    assert shader.image.channels[0].renderpass.buffer_idx == "a"
    assert shader.image.channels[1].renderpass.buffer_idx == "b"
    assert shader.image.channels[2].renderpass.buffer_idx == "c"
    assert shader.image.channels[3].renderpass.buffer_idx == "d"
    assert shader.image.channels[1].sampler_settings["address_mode_u"] == "repeat"
    assert (
        shader.image.channels[3].sampler_settings["address_mode_u"] == "clamp-to-edge"
    )
