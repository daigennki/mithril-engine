#include "smaa_common.glsl"

layout(binding = 0) uniform texture2D area_tex;
layout(binding = 1) uniform texture2D search_tex;
layout(binding = 2) uniform texture2D edgesTex;

layout(location = 0) uniform in vec2 texcoord;
layout(location = 1) uniform in vec2 pixcoord;
layout(location = 2) uniform in vec4 offset[3];

layout(location = 0) out vec4 color_out;

void main()
{
	color_out = SMAABlendingWeightCalculationPS(texcoord, pixcoord, offset, edges_tex, area_tex, search_tex, 0);
}
