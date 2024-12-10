from wgpu_shadertoy import BufferRenderPass, Shadertoy, ShadertoyChannelBuffer

# shadertoy source: https://www.shadertoy.com/view/lljcDG by rkibria CC-BY-NC-SA-3.0
image_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    vec3 col = texture( iChannel0, uv ).xyz;
    // col += sin(iTime);
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
    
    fragColor = vec4(k, j, 0.0, 1.0);
}
"""

# theoretically we can use a buffer pass in wgsl as well, and mix them. Just set shader_type = "wgsl"!
buffer_code_wgsl = """
fn shader_main(frag_coord: vec2<f32>) -> vec4<f32>{
    let uv = frag_coord / i_resolution.xy;
    let col = textureSample(i_channel0, sampler0, uv).xyz;

    var k = col.x;
    var j = col.y;

    let inc = ((uv.x + uv.y) / 100.0) * 0.99 + 0.01;

    if (j == 0.0) {
        k += inc;
    }
    else {
        k -= inc;
    }
    
    if (k >= 1.0){
        j = 1.0;
    }
        
    if (k <= 0.0){
        j = 0.0;
    
    }
    return vec4<f32>(k, j, 0.0, 1.0);
}

"""


buffer_a_channel = ShadertoyChannelBuffer(buffer="a", wrap="repeat")
# using the wgsl translated code for the buffer pass
buffer_a_pass_wgsl = BufferRenderPass(
    buffer_idx="a", code=buffer_code_wgsl, inputs=[buffer_a_channel], shader_type="wgsl"
)
shader = Shadertoy(
    image_code,
    inputs=[buffer_a_channel],
    buffers={"a": buffer_a_pass_wgsl},
    profile=True,
)

if __name__ == "__main__":
    shader.show()
