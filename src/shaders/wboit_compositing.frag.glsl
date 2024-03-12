#version 460

// The shader used to composite transparent objects over opaque objects, when using WBOIT/MBOIT.

#ifdef MULTISAMPLED_IMAGE
#define IMAGE_TYPE image2DMS
#else
#define IMAGE_TYPE image2D
#endif

/* sum(rgb * a, a) */
layout(binding = 0, rgba16f) uniform readonly IMAGE_TYPE accum_texture;

/* prod(1 - a) */
layout(binding = 1, r8) uniform readonly IMAGE_TYPE revealage_texture;

layout(location = 0) out vec4 color_out;

void main()
{
	ivec2 load_coord = ivec2(gl_FragCoord.xy);

#ifdef MULTISAMPLED_IMAGE
	// Use the sample with the minimum revealage so that we don't run the shader for every sample.
	int num_samples = imageSamples(revealage_texture);
	int min_revealage_sample = 0;
	float revealage = 1.0;
	for (int i = 0; i < num_samples; i += 1) {
		float sample_revealage = imageLoad(revealage_texture, load_coord, i).r;
		min_revealage_sample = (sample_revealage < revealage ? i : min_revealage_sample);
		revealage = (min_revealage_sample == i ? sample_revealage : revealage);
	}
	vec4 accum = imageLoad(accum_texture, load_coord, min_revealage_sample);
#else
	vec4 accum = imageLoad(accum_texture, load_coord);
	float revealage = imageLoad(revealage_texture, load_coord).r;
#endif

	accum = min(accum, 16777216); // suppress overflow

	vec3 average_color = accum.rgb / max(accum.a, 0.00001);
	color_out = vec4(average_color, revealage);
}

