from wgpu_shadertoy import Shadertoy

# shadertoy source: https://www.shadertoy.com/view/Wf3SWn by Xor CC-BY-NC-SA 3.0
# modified in Line73 to disassemble the for loop due to: https://github.com/gfx-rs/wgpu/issues/6208

shader_code = """//glsl
/*
    "Sunset" by @XorDev
    
    Expanded and clarified version of my Sunset shader:
    https://www.shadertoy.com/view/wXjSRt
    
    Based on my tweet shader:
    https://x.com/XorDev/status/1918764164153049480
*/

//Output image brightness
#define BRIGHTNESS 1.0

//Base brightness (higher = brighter, less saturated)
#define COLOR_BASE 1.5
//Color cycle speed (radians per second)
#define COLOR_SPEED 0.5
//RGB color phase shift (in radians)
#define RGB vec3(0.0, 1.0, 2.0)
//Color translucency strength
#define COLOR_WAVE 14.0
//Color direction and (magnitude = frequency)
#define COLOR_DOT vec3(1,-1,0)

//Wave iterations (higher = slower)
#define WAVE_STEPS 8.0
//Starting frequency
#define WAVE_FREQ 5.0
//Wave amplitude
#define WAVE_AMP 0.6
//Scaling exponent factor
#define WAVE_EXP 1.8
//Movement direction
#define WAVE_VELOCITY vec3(0.2)


//Cloud thickness (lower = denser)
#define PASSTHROUGH 0.2

//Cloud softness
#define SOFTNESS 0.005
//Raymarch step
#define STEPS 100.0
//Sky brightness factor (finicky)
#define SKY 10.0
//Camera fov ratio (tan(fov_y/2))
#define FOV 1.0

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    //Raymarch depth
    float z = 0.0;
    
    //Step distance
    float d = 0.0;
    //Signed distance
    float s = 0.0;
    
    //Ray direction
    vec3 dir = normalize( vec3(2.0*fragCoord - iResolution.xy, - FOV * iResolution.y));
    
    //Output color
    vec3 col = vec3(0);
    
    //Clear fragcolor and raymarch with 100 iterations
    for(float i = 0.0; i<STEPS; i++)
    {
        //Compute raymarch sample point
        vec3 p = z * dir;
        
        //Turbulence loop
        //https://www.shadertoy.com/view/3XXSWS
        float j, f = WAVE_FREQ;
        for(j = 0.0; j<WAVE_STEPS; j++) {
            p += WAVE_AMP*sin(p*f - WAVE_VELOCITY*iTime).yzx / f;
            f *= WAVE_EXP;
        }
        //Compute distance to top and bottom planes
        s = 0.3 - abs(p.y);
        //Soften and scale inside the clouds
        d = SOFTNESS + max(s, -s*PASSTHROUGH) / 4.0;
        //Step forward
        z += d;
        //Coloring with signed distance, position and cycle time
        float phase = COLOR_WAVE * s + dot(p,COLOR_DOT) + COLOR_SPEED*iTime;
        //Apply RGB phase shifts, add base brightness and correct for sky
        col += (cos(phase - RGB) + COLOR_BASE) * exp(s*SKY) / d;
    }
    //Tanh tonemapping
    //https://www.shadertoy.com/view/ms3BD7
    col *= SOFTNESS / STEPS * BRIGHTNESS;
    fragColor = vec4(tanh(col * col), 1.0);
}
"""

shader = Shadertoy(shader_code, resolution=(800, 450), imgui=True)


if __name__ == "__main__":
    shader.show()