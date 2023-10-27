[[vk::push_constant]] cbuffer transformation
{
    float4x4 projviewmodel;	// pre-multiplied projection, view, and model transformation matrices
};
float4 main(float3 position : POSITION) : SV_POSITION
{
	return mul(projviewmodel, float4(position, 1.0));
}
