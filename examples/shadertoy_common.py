from wgpu_shadertoy import Shadertoy

# shader_source https://www.shadertoy.com/view/XcSXWD by Vipitis

common_code = """
// degree of red from 0.0 to 1.0 as a function
vec3 getRed(float r){
    return vec3(r, 0.0, 0.0);
}

// solid green as a variable
vec3 green = vec3(0.0, 1.0, 0.0);
"""

shader_code = """
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord - .5/iResolution.xy)/iResolution.y;

    vec3 col = getRed(fract(iTime));
    col += green;
    // Output to screen
    fragColor = vec4(col,1.0);
}
"""

shader = Shadertoy(shader_code, common=common_code, resolution=(800, 450))

if __name__ == "__main__":
    shader.show()
