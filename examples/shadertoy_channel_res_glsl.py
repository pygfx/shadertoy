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

img_data = np.array(Image.open("./examples/screenshots/shadertoy_star.png"))
channel0 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 0)), wrap="clamp")
channel1 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 1)), wrap="clamp")
channel2 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 2)), wrap="repeat")
channel3 = ShadertoyChannel(np.ascontiguousarray(np.rot90(img_data, 3)), wrap="repeat")
shader = Shadertoy(
    shader_code, resolution=(1200, 900), inputs=[channel0, channel1, channel2, channel3]
)

if __name__ == "__main__":
    shader.show()
