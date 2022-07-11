SamplerState sampler0 : register(s0, space1);
TextureCube skybox : register(t0, space1);
float4 main(float3 frag_pos : POSITION) : SV_TARGET
{
	return skybox.Sample(sampler0, frag_pos.xzy);
}