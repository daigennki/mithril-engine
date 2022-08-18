SamplerState sampler0 : register(s0, space2);
Texture2D base_color : register(t1, space2);
struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};
float4 main(PS_INPUT input) : SV_Target
{
    float4 texColor = base_color.Sample(sampler0, input.uv);
    texColor.rgb *= texColor.a;
	return texColor;
}

