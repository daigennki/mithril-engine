cbuffer model : register(b0)
{
	float4x4 transform;	    // rotation, scale, and translation
};
cbuffer projviewmat : register(b0, space1)
{
    float4x4 projview;
	float4x4 proj;
	float4x4 view;
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
	//float3 normal : NORMAL;
};
VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT output;
    output.pos = mul(transform, float4(input.pos, 1.0));
    output.pos = mul(projview, output.pos);
    output.uv = input.uv;
	float3x3 transform3 = float3x3(transform[0].xyz, transform[1].xyz, transform[2].xyz);
	//output.normal = mul(transform3, input.normal);
	return output;
}
