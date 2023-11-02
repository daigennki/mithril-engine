#include "smaa_common.glsl"

layout(binding = 0) uniform texture2D shaded_output;

layout(location = 0) in vec2 texcoord;
layout(location = 1) in vec4 offset[3];

layout(location = 0) out vec2 color_out;

void main()
{
    color_out = SMAALumaEdgeDetectionPS(texcoord, offset, shaded_output);
}
