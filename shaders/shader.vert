#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 color;

out vec4 vColor;

/*
 * Converts HSV value to RGB
 * Code from http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
 * Running in vertex shader over fragment shader for performance
 */
vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    gl_Position = vec4(position, 0.0, 1.0);

    vec3 hsvColor = vec3(color, 1.0);
    vec3 rgbColor = hsv2rgb(hsvColor);
    vColor = vec4(rgbColor, 1.0);
}
