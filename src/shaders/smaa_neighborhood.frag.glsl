#version 460
#define SMAA_INCLUDE_PS 1
#include "smaa_common.glsl"

layout(binding = 1) uniform sampler2D shaded_output0;
layout(binding = 2) uniform sampler2D shaded_output1;
layout(binding = 3) uniform sampler2D blend_tex0;
layout(binding = 4) uniform sampler2D blend_tex1;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec4 offset;

layout(location = 0) out vec4 color_out;

void main()
{
	vec4 color0 = SMAANeighborhoodBlendingPS(texcoord, offset, shaded_output0, blend_tex0);

#ifdef S2X
	vec4 color1 = SMAANeighborhoodBlendingPS(texcoord, offset, shaded_output1, blend_tex1);
	color_out = mix(color0, color1, 0.5);
#else
	color_out = color0;
#endif
}
