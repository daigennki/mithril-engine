// The shader code used for moment-based OIT moment writes (stage 2).

SamplerState sampler0 : register(s0);
Texture2D base_color : register(t1);

struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float2 uv : TEXCOORD;
};
struct PS_OUTPUT
{
	float4 moments : SV_Target0;
	float optical_depth : SV_Target1;
	float min_depth : SV_Target2;
};

float depth_to_unit(float z, float c0, float c1)
{
	return log(z * c0) * c1;
}
float4 make_moments4(float z)
{
	float zsq = z * z;
	return float4(z, zsq, zsq * z, zsq * zsq);
}
PS_OUTPUT main(PS_INPUT input)
{
	float alpha = base_color.Sample(sampler0, input.uv).a;
	float depth = input.pos.z;

	// TODO: use a descriptor set here to reflect changes to the camera near/far planes
	const float near = 0.25;
	const float far = 5000.0;
	const float c0 = 1.0 / near;
	const float c1 = 1.0 / log(far / near);

	const float k_max_alpha = 1.0 - 0.5 / 256.0;
	float optical_depth = -log(1.0 - (alpha * k_max_alpha));
	float unit_pos = depth_to_unit(depth, c0, c1);

	PS_OUTPUT output;
	output.moments = make_moments4(unit_pos) * optical_depth;
	output.optical_depth = optical_depth;
	output.min_depth = depth;
	return output;
}

