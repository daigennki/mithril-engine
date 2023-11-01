layout(binding = 0) uniform smaa_metrics
{
    vec4 SMAA_RT_METRICS;
};
#define SMAA_AREATEX_SELECT(sample) sample.gr
#define SMAA_GLSL_4
#define SMAA_PRESET_ULTRA
#include "SMAA.glsl"
