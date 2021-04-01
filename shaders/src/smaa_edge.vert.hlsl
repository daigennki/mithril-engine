#include "smaa_common.hlsl"

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texcoords : TEXCOORD0;
    float4 offset[3] : TEXCOORD1;
};
VS_OUTPUT main(uint vid : SV_VertexID)
{
    VS_OUTPUT output;
    float2 texcoords[3] = { { 0.0, 0.0 }, { 2.0, 0.0 }, { 0.0, 2.0 } };
    output.texcoords = texcoords[min(vid,2)];
    output.position = float4(mad(output.texcoords, 2.0, -1.0), 0.0, 1.0);
    SMAAEdgeDetectionVS(output.texcoords, output.offset);
    return output;
}