SamplerState sampler0 : register(s2);
Texture2D tex : register(t1);
struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
    //float4 color : COLOR;
};
float4 main(PS_INPUT input) : SV_Target
{
    float4 texColor = tex.Sample(sampler0, input.uv);
    //texColor *= input.color;
    texColor.rgb *= texColor.a;
	return texColor;
}