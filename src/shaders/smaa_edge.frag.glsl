#version 460
#define SMAA_INCLUDE_PS 1
#include "smaa_common.glsl"

layout(binding = 1) uniform sampler2D shaded_output;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec4 offset[3];

layout(location = 0) out vec2 color_out;

void main()
{
    color_out = SMAALumaEdgeDetectionPS(texcoord, offset, shaded_output);
}
