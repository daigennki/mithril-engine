SamplerState sampler0 : register(s0);
Texture2D tex_y : register(t0);
Texture2D tex_c : register(t1);
struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
};

// Convert from Rec.709 to linear gamma
float gammaRec709ToLinear(float channel)
{
    /*if (channel >= 0.018) return pow(channel / 1.099 + 0.099 / 1.099, 1.0 / 0.45);
    else return channel / 4.5;*/
    return pow(channel, 1.0 / 0.45);    // Rec.709 "conceptual" gamma, appears closer to how it does in VLC
}

float4 main(PS_INPUT input) : SV_Target
{
    // Rec.709 color matrix
    float3x3 colorMatrix = float3x3(
        1.0, 0.0,    1.5748,
        1.0, -0.1873, -0.4681,
        1.0, 1.8556,  0.0
    );
    float3 conv;
    conv.r = tex_y.Sample(sampler0, input.uv).r;
    conv.gb = tex_c.Sample(sampler0, input.uv).rg;
    conv.gb = (conv.gb - 128.0 / 255.0) * (219.0 / 224.0);

    // color space conversion
    conv = mul(colorMatrix, conv);

    // limited range to full range RGB
    conv.rgb = (conv.rgb - 16.0 / 255.0) * (255.0 / 219.0);

    // gamma correction
    conv.r = gammaRec709ToLinear(conv.r);
    conv.g = gammaRec709ToLinear(conv.g);
    conv.b = gammaRec709ToLinear(conv.b);

	return input.color * float4(conv, 1.0);
}