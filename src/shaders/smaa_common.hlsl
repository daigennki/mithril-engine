cbuffer smaaMetrics : register(b0)
{
    float4 SMAA_RT_METRICS;
};
#define SMAA_AREATEX_SELECT(sample) sample.gr
#define SMAA_HLSL_4_1
#define SMAA_PRESET_ULTRA
#include "SMAA.hlsl"