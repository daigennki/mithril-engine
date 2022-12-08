[[vk::push_constant]] cbuffer renderParams
{
	float4x4 sky_projview;
};
struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float3 frag_pos : POSITION; //vertex position to fragment shader
};
VS_OUTPUT main(float3 position : POSITION)
{
    VS_OUTPUT output;
    output.frag_pos = position;
	float4 new_pos = mul(sky_projview, position);
	output.position = new_pos.xyww;
    return output;
}
