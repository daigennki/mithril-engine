#version 460
#define SMAA_INCLUDE_PS 1
#include "smaa_common.glsl"

layout(binding = 1) uniform sampler2D shaded_output0;
layout(binding = 2) uniform sampler2D blend_tex0;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec4 offset;

layout(location = 0) out vec4 color_out;

void main()
{
	color_out = SMAANeighborhoodBlendingPS(texcoord, offset, shaded_output0, blend_tex0);
}
