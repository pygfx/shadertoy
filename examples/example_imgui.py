from wgpu_shadertoy import Shadertoy

# shadertoy source: https://www.shadertoy.com/view/dllyzH by timmaffett at CC-BY-NC-SA 3.0 (


shader_code = """
// Fork of "圆形烟花" by houkinglong. https://shadertoy.com/view/dlsyRr
// 2023-07-27 05:32:56

#define HARDNESS 60.0
#define AMOUNT 90
#define MAX_DISTANCE 20.0
#define SPEED 0.15

#define PI  3.14159265359
#define TAU 6.28318530717
vec3 hsb2rgb( in vec3 c )
{
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),
                             6.0)-3.0)-1.0,
                     0.0,
                     1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return (c.z * mix( vec3(1.0), rgb, c.y));
}


// https://iquilezles.org/articles/smin
float smin(float a, float b, float k) {
    float res = exp2(-k * a) + exp2(-k * b);
    return -log2(res) / k;
}

float sdCircle(vec2 uv, vec2 pos, float radius) {
    return length(uv - pos) - radius;
}

float sdLine(vec2 uv, vec2 start, vec2 end) {
    return 0.0;
}

float randomSingle(vec2 p) {
    p = fract(p * vec2(233.34, 851.73));
    p += dot(p, p + 23.45);
    return fract(p.x * p.y);
}

vec4 randomPoint(vec2 p) {
    float x = randomSingle(p);
    float y = randomSingle(vec2(x, p.x));
    return vec4(x, y, randomSingle(vec2(y, x)), randomSingle(vec2(x, y)));
}

float Star(vec2 uv, float dist, vec2 id) {
    vec4 rand = randomPoint(id);

    float progress = fract(iTime * SPEED + rand.z);

    vec2 dir = 2.0 * (normalize(rand.xy) - 0.5);

    rand.w = clamp((rand.w - 0.5) * 999.0, -1.0, 1.0);
    dir *= rand.w;

    return smin(dist, sdCircle(uv, dir * progress * MAX_DISTANCE, 0.001) / (progress + 0.7), 200.0);
}

float Graph(vec2 uv, float r) {
    float dist = sdCircle(uv, vec2(0.0, 0.0), r);

    dist = Star(uv, dist, vec2(1.0, 1.0));

    for (int s = 1; s < AMOUNT; ++s)
        dist = Star(uv, dist, vec2(-1.0, s));

    dist *= HARDNESS;

    dist = max(dist, 0.0);
    dist = 1.0 / (dist + 0.001);
    dist *= clamp(0.8, 0.98, length(uv));

    return dist;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;


    float angle = iTime * 0.4;

    // compute angle for pixel and to hsv color
    vec2 center = iResolution.xy /2.0;
    vec2 toRing = normalize(fragCoord.xy - center);
    float ringValue = atan(toRing.x, -toRing.y);
    ringValue /= PI;
    ringValue = 1. - ringValue;
    float ringAngle = ringValue / 2.0; // scale to hsv func
    vec3 hsv = hsb2rgb(vec3(ringAngle+angle, 0.85, 0.96));
    
    // 一个像素的大小
    float scale = 1.0 / iResolution.y;

    // 外圆半径
    float outerRadius = 0.99;
    // 内圆半径
    float innerRadius = 0.3;

    // 色值声明
    vec3 col = vec3(0);
    //vec3 colR = vec3(1.0, 0.0, 0.0);
    //vec3 colG = vec3(0.0, 0.3, 0.0);
    vec3 colB = vec3(0.2);//vec3(0.01, 0.13, 0.32);// hsv*0.1;//rotate outer ring color

    float dis = length(uv);

    // 圆底
    float bg = smoothstep(scale, -scale, sdCircle(uv, vec2(0.0, 0.0), outerRadius));
    col = mix( vec3(0.0), colB, bg);

    // 渐变环
    float ring = smoothstep(outerRadius - 0.2, outerRadius, dis);
    col *= ring;

    if (dis > outerRadius) {
        fragColor = vec4(hsv,1.) * 0.1;//vec4(0.0);
        return;
    }



#define COLORCENTER
#ifdef COLORCENTER
     if (dis < innerRadius/2.9) {
         fragColor = vec4(hsb2rgb(vec3(ringAngle+angle, 0.85, 0.96)),1.0);//vec4(0.0, 0.1, 0.2, 1.);
         return;
     }
#endif

    // 增加旋转
    float sinA = sin(angle);
    float cosA = cos(angle);
    mat2 rot = mat2(cosA, -sinA, sinA, cosA);
    uv *= rot;

    uv *= 3.0;

    float m = Graph(uv, innerRadius);

    vec3 tint = hsb2rgb(vec3(ringAngle+angle, 0.85, 0.96));//hsv;//vec3(0.0, 0.0, 1.0);

    col += m * mix(tint, tint*0.8/*vec3(1.0, 1.0, 1.0)*/, m);

    fragColor = vec4(col, 1.0);
}
"""

shader = Shadertoy(shader_code, resolution=(800, 450), imgui=True)


if __name__ == "__main__":
    shader.show()