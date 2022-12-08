cbuffer model : register(b0)
{
	float4x4 transform;	    // rotation, scale, and translation
};
[[vk::push_constant]] cbuffer projviewmat
{
    float4x4 projview;
};
struct VS_INPUT
{
    float3 pos : POSITION;
    float2 uv : TEXCOORD;
	float3 normal : NORMAL;
};
struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
	float2 uv : TEXCOORD;
};
VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    output.pos = mul(transform, float4(input.pos, 1.0));
    output.pos = mul(projview, output.pos);
    output.uv = input.uv;
	return output;
}
