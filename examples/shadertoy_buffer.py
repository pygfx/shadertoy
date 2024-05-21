# run_example = false
# buffer passes in development
from wgpu_shadertoy import BufferRenderPass, Shadertoy
from wgpu_shadertoy.inputs import ShadertoyChannelBuffer

# shadertoy source: https://www.shadertoy.com/view/lljcDG by rkibria CC-BY-NC-SA-3.0
image_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    vec3 col = texture( iChannel0, uv ).xyz;
	fragColor = vec4(col,1.0);
}
"""

buffer_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    vec3 col = texture( iChannel0, uv ).xyz;

    float k = col.x;
    float j = col.y;

	float inc = ((uv.x + uv.y) / 100.0) * 0.99 + 0.01;

    if (j == 0.0) {
	    k += inc;
    }
    else {
	    k -= inc;
    }
    
    if (k >= 1.0)
        j = 1.0;

    if (k <= 0.0)
        j = 0.0;
    
    fragColor = vec4(0.5, j, 0.0, 1.0);
}
"""

buffer_a_channel = ShadertoyChannelBuffer(buffer="a", wrap="repeat")
buffer_a_pass = BufferRenderPass(
    buffer_idx="a", code=buffer_code, inputs=[buffer_a_channel]
)
shader = Shadertoy(
    image_code,
    inputs=[buffer_a_channel],
    buffers={"a": buffer_a_pass},
    resolution=(512, 256),
)
if __name__ == "__main__":
    shader.show()
