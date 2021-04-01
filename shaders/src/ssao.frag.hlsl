struct VS_OUTPUT
{
    float4 position : SV_POSITION;
    float2 texcoords : TEXCOORD;
};

cbuffer renderParams : register(b0)
{
	float4x4 proj;
    float4x4 view;	// view matrix
    float4x4 inverseProjView;
    float3 camera_pos;
    uint rpFiller;
	uint msCount;
};
cbuffer ssaoParams : register(b0, space2)
{
    float2 noiseScale;
    float ssao_bias;
    float radius;
    float3 samples[16];
};
Texture2D gDepth : register(t0, space1);
Texture2D gNormal : register(t2, space1);
Texture2D texNoise : register(t0, space2);
SamplerState texNoiseSampler : register(s0, space2);

float3 getFragPos(float2 where, int2 texel_where, int ms_sample)
{
    // Reconstruct view space position
    float fragDepth = gDepth.Load(int3(texel_where, ms_sample)).r;
    float4 ReconstructedPos = float4(mad(where, 2.0, -1.0), fragDepth, 1.0);
    ReconstructedPos = mul(view, mul(inverseProjView, ReconstructedPos));
    return ReconstructedPos.xyz / ReconstructedPos.w;
}
float main(VS_OUTPUT input) : SV_Target
{
    int2 texSize = int2(0,0);
    gNormal.GetDimensions(texSize.x, texSize.y);
    int2 texelPos = int2(texSize * input.texcoords);
    float3 randomVec = normalize(texNoise.Sample(texNoiseSampler, input.texcoords * noiseScale).xyz);

    // going through all samples is expensive so we use only one random sample
    int s = 0;//int(round(randomVec.x * (msCount - 1)));

    float3 fragPos = getFragPos(input.texcoords, texelPos, s).xyz;
    float3 normal = gNormal.Load(int3(texelPos, s)).rgb;
    float3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    float3 bitangent = cross(normal, tangent);
    float3x3 tbn = float3x3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < 16; ++i)
    {
        float3 occlusionSample = fragPos + mul(tbn, samples[i]);
        float4 offset = float4(occlusionSample, 1.0);
        offset = mul(proj, offset);
        offset /= offset.w;
        offset.xy = mad(offset.xy, 0.5, 0.5);
        float sampleDepth = getFragPos(offset.xy, int2(offset.xy * texSize), 0).z;
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth <= occlusionSample.z + ssao_bias ? 1.0 : 0.0) * rangeCheck;
    }
    occlusion = mad(occlusion, -1.0 / 16.0, 1.0);
    return occlusion;
}