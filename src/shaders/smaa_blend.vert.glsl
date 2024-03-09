#version 460
#define SMAA_INCLUDE_VS 1
#include "smaa_common.glsl"

layout(location = 0) out vec2 texcoord;
layout(location = 1) out vec2 pixcoord;
layout(location = 2) out vec4 offset[3];

void main()
{
    vec2 texcoords[3] = { { 0.0, 0.0 }, { 2.0, 0.0 }, { 0.0, 2.0 } };
    texcoord = texcoords[min(gl_VertexIndex,2)];
    gl_Position = vec4(texcoord * 2.0 - 1.0, 0.0, 1.0);

    SMAABlendingWeightCalculationVS(texcoord, pixcoord, offset);
}
