cbuffer transform_ubo : register(b0)
{
	float4x4 transformation;
};
struct VS_INPUT
{
    float2 pos : POSITION;
    float2 uv : TEXCOORD;
};
struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};
VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    output.pos = mul(transformation, float4(input.pos, 0.0, 1.0));
	output.uv = input.uv;
    return output;
}