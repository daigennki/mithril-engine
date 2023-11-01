#include "smaa_common.glsl"

layout(location = 0) out vec2 texcoord;
layout(location = 1) out vec4 offset[3];

void main()
{
    vec2 texcoords[3] = { { 0.0, 0.0 }, { 2.0, 0.0 }, { 0.0, 2.0 } };
    texcoord = texcoords[min(gl_VertexIndex,2)];
    gl_Position = vec4(texcoord * 2.0 - 1.0, 0.0, 1.0);

    SMAAEdgeDetectionVS(texcoord, offset);
}
