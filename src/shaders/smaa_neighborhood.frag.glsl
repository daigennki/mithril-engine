#include "smaa_common.glsl"

layout(binding = 0) texture2D shaded_output : register(t0);
layout(binding = 1) texture2D blend_tex : register(t1);

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec4 offset;

layout(location = 0) out vec4 color_out;

void main()
{
    color_out = SMAANeighborhoodBlendingPS(texcoord, offset, shaded_output, blend_tex);
}
