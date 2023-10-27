[[vk::push_constant]] cbuffer transformation
{
    float4x4 projview;	// pre-multiplied projection and view matrices
	float4x4 model_transform;	// just the model transformation matrix
};
struct PointLight 
{
	float4 position;
	float4 direction;
    float3 color;
    float range;
};
cbuffer cur_light : register(b0)
{
	PointLight light;
};

float4 main(float3 position : POSITION) : SV_POSITION
{
	float4 output = mul(model_transform, float4(position, 1.0));
	float dist = length(output.xyz - light.position.xyz);	// get distance between fragment and light source
    output = mul(projview, output);
	output.z = output.w * dist / (light.range * 2);	// map to [0;1] range by dividing by far plane
	return output;
}
