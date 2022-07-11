#include "smaa_common.hlsl"

Texture2D areaTex : register(t0);
Texture2D searchTex : register(t1);
Texture2D edgesTex : register(t2);

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texcoords : TEXCOORD0;
    float2 pixcoord : TEXCOORD1;
    float4 offset[3] : TEXCOORD2;
};
float4 main(VS_OUTPUT input) : SV_TARGET
{
    return SMAABlendingWeightCalculationPS(input.texcoords, input.pixcoord, input.offset, edgesTex, areaTex, searchTex, 0);
}