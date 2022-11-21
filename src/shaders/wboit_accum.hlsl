
// The shader code used for WBOIT accumulation, meant to be included by other shaders.

struct PS_OUTPUT
{
	float4 accum : SV_Target0;
	float revealage : SV_Target1;
};

PS_OUTPUT write_transparent_pixel(float4 premul_reflect, float depth, float2 screen_pos) 
{ 
    /* Modulate the net coverage for composition by the transmission. This does not affect the color channels of the
       transparent surface because the caller's BSDF model should have already taken into account if transmission modulates
       reflection. This model doesn't handled colored transmission, so it averages the color channels. See 

          McGuire and Enderton, Colored Stochastic Shadow Maps, ACM I3D, February 2011
          http://graphics.cs.williams.edu/papers/CSSM/

       for a full explanation and derivation.*/
    premul_reflect.a *= 1.0 - clamp((premul_reflect.r + premul_reflect.g + premul_reflect.b) * (1.0 / 3.0), 0, 1);

    /* You may need to adjust the w function if you have a very large or very small view volume; see the paper and
       presentation slides at http://jcgt.org/published/0002/02/09/ */
    // Intermediate terms to be cubed
    float a = mad(min(1.0, premul_reflect.a), 8.0, 0.01);
    float b = mad(depth, -0.95, 1.0);
    float w = clamp(a * a * a * 1e8 * b * b * b, 1e-2, 3e2);

	PS_OUTPUT output;
	output.accum = premul_reflect * w;
    output.revealage = premul_reflect.a;
	return output;
}

