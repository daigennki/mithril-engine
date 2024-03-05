// The shader code used for WBOIT accumulation, meant to be included by other shaders.

layout(location = 0) out vec4 accum;
layout(location = 1) out float revealage;

void write_transparent_pixel(vec4 premul_reflect)
{
	// Implementation of equation 10 (one of the "generic weight functions") in the WBOIT paper
	float w = premul_reflect.a * max(1e-2, 3e3 * pow(1.0 - gl_FragCoord.z, 3));
	accum = premul_reflect * w;
    revealage = premul_reflect.a;
}

