[[vk::push_constant]] cbuffer projviewmodel
{
    float4x4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
	float4x4 transform_notranslate;	// the model transformation matrix with just scale and rotation
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
    output.pos = mul(projviewmodel, float4(input.pos, 1.0));
    output.uv = input.uv;
	return output;
}
