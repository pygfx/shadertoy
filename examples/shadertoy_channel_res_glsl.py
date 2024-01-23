import numpy as np
from PIL import Image

from wgpu_shadertoy import Shadertoy, ShadertoyChannel

# shadertoy source: https://www.shadertoy.com/view/4f2SzR by Vipitis

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

img_data = np.array(Image.open("./examples/screenshots/shadertoy_star.png"))
channel0 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 0)), wrap="clamp")
channel1 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 1)), wrap="clamp")
channel2 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 2)), wrap="repeat")
channel3 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 3)), wrap="repeat")
shader = Shadertoy(
    shader_code_wgsl,
    resolution=(1200, 900),
    inputs=[channel0, channel1, channel2, channel3],
)

if __name__ == "__main__":
    shader.show()
