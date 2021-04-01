cbuffer model_pvm : register(b0)
{
	float4x4 transform;	// rotation, scale, and translation
};
struct PointLight 
{
	float4 position;
	float4 direction;
    float3 color;
    float range;
};
cbuffer curLightUBO : register(b1, space1)
{
	PointLight curLight;
};
cbuffer projviewmat : register(b0, space1)
{
	float4x4 projview;
};
float4 main(float3 position : POSITION) : SV_POSITION
{
	float4 output = mul(transform, float4(position, 1.0));
	float dist = length(output.xyz - curLight.position.xyz);	// get distance between fragment and light source
    output = mul(projview, output);
	output.z = output.w * dist / (curLight.range * 2);	// map to [0;1] range by dividing by far plane
	return output;
}