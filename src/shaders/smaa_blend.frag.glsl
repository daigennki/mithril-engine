#version 460
#define SMAA_INCLUDE_PS 1
#include "smaa_common.glsl"

layout(binding = 1) uniform sampler2D area_tex;
layout(binding = 2) uniform sampler2D search_tex;
layout(binding = 3) uniform sampler2D edges_tex;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec2 pixcoord;
layout(location = 2) in vec4 offset[3];

layout(location = 0) out vec4 color_out;

void main()
{
	color_out = SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset, edges_tex, area_tex, search_tex, vec4(0.0));
}
