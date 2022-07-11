cbuffer model : register(b0)
{
	float4x4 transform;	// rotation, scale, and translation
};
cbuffer projviewmat : register(b0, space1)
{
	float4x4 projview;
};
float4 main(float3 position : POSITION) : SV_POSITION
{
    float4 FragPos = mul(transform, float4(position, 1.0));
	return mul(projview, FragPos);
}