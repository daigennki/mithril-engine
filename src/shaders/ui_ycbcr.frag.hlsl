SamplerState sampler0 : register(s0);
Texture2D tex_y : register(t0);
Texture2D tex_cb : register(t1);
Texture2D tex_cr : register(t2);
struct PS_INPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
};

// Convert from Rec.709 to linear gamma
//
// for 1 >= L >= 0.018
// 1.099L^0.45 = V + 0.099
// L^0.45 = V / 1.099 + 0.099 / 1.099
// L = (V / 1.099 + 0.099 / 1.099) ^ (1.0 / 0.45)
//
// for 0.018 > L >= 0
// 4.5L = V
// L = V / 4.5
float gammaRec709ToLinear(float channel)
{
    /*if (channel >= 0.018) return pow(channel / 1.099 + 0.099 / 1.099, 1.0 / 0.45);
    else return channel / 4.5;*/
    return pow(channel, 1.0 / 0.45);    // Rec.709 "conceptual" gamma, appears closer to how it does in VLC
}

float4 main(PS_INPUT input) : SV_Target
{
    // Rec.709 color matrix
    // E'Y = 0.2126 E'R + 0.7152 E'G + 0.0722 E'B
    // E'CB = (E'B - E'Y) / 1.8556
    // E'CR = (E'R - E'Y) / 1.5748
    // 
    // E'B - E'Y = 1.8556 E'CB
    // E'B = E'Y + 1.8556 E'CB
    //
    // E'R - E'Y = 1.5748 E'CR
    // E'R = E'Y + 1.5748 E'CR
    //
    // E'Y - 0.7152 E'G = 0.2126 E'R + 0.0722 E'B
    // -0.7152 E'G = -E'Y + 0.2126 E'R + 0.0722 E'B
    // E'G = 1.3982 E'Y - 0.2973 E'R - 0.1010 E'B
    // E'G = 1.3982 E'Y - 0.2973 (E'Y + 1.5748 E'CR) - 0.1010 (E'Y + 1.8556 E'CB)
    // E'G = 1.3982 E'Y - 0.2973 E'Y - 0.4601 E'CR - 0.1010 E'Y - 0.1873 E'CB
    // E'G = 1.3982 E'Y - 0.2973 E'Y - 0.1010 E'Y - 0.1873 E'CB - 0.4681 E'CR
    // E'G = E'Y - 0.1873 E'CB - 0.4681 E'CR
    float3x3 colorMatrix = float3x3(
        1.0, 0.0,    1.5748,
        1.0, -0.1873, -0.4681,
        1.0, 1.8556,  0.0
    );
    float3 conv;
    conv.r = tex_y.Sample(sampler0, input.uv).r;
    conv.g = tex_cb.Sample(sampler0, input.uv).r;
    conv.b = tex_cr.Sample(sampler0, input.uv).r;
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