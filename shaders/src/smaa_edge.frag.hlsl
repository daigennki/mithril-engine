#include "smaa_common.hlsl"

Texture2D shadedOutput : register(t0, space1);

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texcoords : TEXCOORD0;
    float4 offset[3] : TEXCOORD1;
};
float2 main(VS_OUTPUT input) : SV_TARGET
{
    return SMAALumaEdgeDetectionPS(input.texcoords, input.offset, shadedOutput);
}