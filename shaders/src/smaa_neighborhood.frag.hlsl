#include "smaa_common.hlsl"

Texture2D shadedOutput : register(t0, space1);
Texture2D shadedOutputDepth : register(t1, space1);
Texture2D blendTex : register(t3);

struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texcoords : TEXCOORD0;
    float4 offset : TEXCOORD1;
};
struct PS_OUTPUT
{
    float4 color : SV_TARGET;
    float depth : SV_DEPTH;
};
PS_OUTPUT main(VS_OUTPUT input)
{
    int2 outputSize;
    shadedOutputDepth.GetDimensions(outputSize.x, outputSize.y);

    PS_OUTPUT output;
    output.color = SMAANeighborhoodBlendingPS(input.texcoords, input.offset, shadedOutput, blendTex);
    output.depth = shadedOutputDepth.Load(int3(outputSize * input.texcoords,0)).r;
    return output;
}