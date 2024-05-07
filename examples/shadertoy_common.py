from wgpu_shadertoy import Shadertoy

# shader source https://www.shadertoy.com/view/4l2BW3 by iq
image_code = """
// The MIT License
// Copyright © 2018 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


// This shader shows how to use the Common tab in Shadertoy: all tabs inherit the
// code in the "Common" tab. In this case, that's used to get the Image and the
// Sound shader to reuse the same music pattern to display the visuals and produce
// the sound.


float shape( in vec2 p, in float n, in float w, in float a )
{    
    float r = length(p);
    float h = 0.7 + 
              (26.0-n)*0.005*
              sin(atan(p.y,p.x)*(2.0+floor(n/4.0))+2.0*iTime)*
              sin((w-50.0)*a*0.2);
    return h - r;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    float n = patternNote( iTime );
    float w = patternFreq( iTime );
    float a = patternFrac( iTime );

    vec2  p = (2.0*fragCoord-iResolution.xy) / iResolution.y;
    
    float e = 2.0/iResolution.y;
    float f  = shape(p,n,w,a);
    float fx = shape(p+vec2(e,0.0),n,w,a);
    float fy = shape(p+vec2(0.0,e),n,w,a);
    
    float d = abs(f) / length( vec2(f-fx,f-fy) );
    float q = smoothstep(1.0,2.0, d );
    q *= 0.8 + 0.2*smoothstep(0.0,10.0, d );

    vec3 col = vec3(1.0,0.8,0.6);
    col.yz += 0.01*sin(p.x+sin(p.y+iTime));
    col *= 1.0 - 0.3*length(p);
    col *= 1.0 - 4.0*a*(1.0-a)*(1.0-q);

    fragColor = vec4(col,1.0);
    
}
"""

# wgpu currently throws error: 'wholeNotes' is invalid The type is not constructible
# so the length of the array as been hardcoded in L74
common_code = """
// The MIT License
// Copyright © 2018 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


// This shader shows how to use the Common tab in Shadertoy: all tabs inherit the
// code in the "Common" tab. In this case, that's used to get the Image and the
// Sound shader to reuse the same music pattern to display the visuals and produce
// the sound.


const float speed = 1.5;

float patternFrac( float x )
{
    return fract(speed*x);
}

const int wholeNotes[7] = int[](0,2,4,5,7,9,11);

float patternNote( float x )
{
    int noteID = int( 7.0+7.0*sin( floor(speed*x) ) );
    return float( wholeNotes[noteID%7] + 12*(noteID/7) );
}

float patternFreq( float x )
{
    float f = patternNote(x);
    return 55.0*pow(2.0,f/12.0);
}
"""

# sound shaders are not yet supported, so this remains unused.
sound_code = """
// The MIT License
// Copyright © 2018 Inigo Quilez
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


// This shader shows how to use the Common tab in Shadertoy: all tabs inherit the
// code in the "Common" tab. In this case, that's used to get the Image and the
// Sound shader to reuse the same music pattern to display the visuals and produce
// the sound.


float tone( in float freq, in float deca, in float time )
{
    // fm sound
    float y = sin(6.2831*freq*time + 5.0*sin(6.2831*freq*time) );
    // add some harmonics
    y *= 0.5*(1.0+y*y);
    // attenaute
    y *= exp(-(1.0+freq/20.0)*deca);
    // attack
    y *= clamp(12.0*deca,0.0,1.0);
	return y;    
}

vec2 mainSound( in int samp, float time )
{
    // reverb
    float y = 0.0;
    float a = 0.7;
    for( int i=0; i<5; i++ )
    {       
        float hime = time - 1.4*float(i)/5.0;
        float freq = patternFreq( hime );
        float deca = patternFrac( hime );
        y += a*tone( freq, deca, hime );
        a *= 0.6;
    }
    
    return vec2( y );
}
"""


shader = Shadertoy(
    image_code, common=common_code, resolution=(800, 450), complete=False
)

if __name__ == "__main__":
    shader.show()
