SamplerState yuvSampler : register(s0);
Texture2D ytex : register(t0);
Texture2D ctex : register(t1);

// Convert from Rec.709 to linear gamma (TODO: figure out why this appears slightly brighter than other programs?)
float gammaRec709ToLinear(float channel)
{
    if (channel < 0.081) return channel * (1.0 / 4.5);
    else return pow(mad(channel, 1.0 / 1.099, 0.099 / 1.099), 1.0 / 0.45);
}
float4 main(float2 texcoords : TEXCOORD) : SV_TARGET
{
    // Rec.709 color matrix
    float3x3 colorMatrix = float3x3(
        1.164, 0.0,    1.793,
        1.164, -0.213, -0.533,
        1.164, 2.112,  0.0
    );
    float3 conv;
    conv.r = ytex.Sample(yuvSampler, texcoords).r;
    conv.gb = ctex.Sample(yuvSampler, texcoords).rg;
    conv -= float3(16.0, 128.0, 128.0) / 255.0;
    conv = mul(colorMatrix, conv);
    conv.r = gammaRec709ToLinear(conv.r);
    conv.g = gammaRec709ToLinear(conv.g);
    conv.b = gammaRec709ToLinear(conv.b);
    return float4(conv, 1.0);
}