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
	// calculate average of accum and revealage across all active samples
	int num_samples = imageSamples(revealage_texture);
	int active_samples = 0;
	vec4 accum = vec4(0.0);
	float revealage = 0.0;
	for (int i = 0; i < num_samples; i += 1) {
		vec4 sample_accum = imageLoad(accum_texture, load_coord, i);
		float sample_revealage = imageLoad(revealage_texture, load_coord, i).r;
		bool sample_active = (sample_revealage != 1.0);
		accum += (sample_active ? sample_accum : vec4(0.0));
		revealage += (sample_active ? sample_revealage : 0.0);
		active_samples += (sample_active ? 1 : 0);
	}
	// only revealage needs to be divided here since accum will be divided later as part of usual
	// WBOIT compositing
	revealage /= active_samples;
#else
	vec4 accum = imageLoad(accum_texture, load_coord);
	float revealage = imageLoad(revealage_texture, load_coord).r;
#endif

	accum = min(accum, 16777216); // suppress overflow

	vec3 average_color = accum.rgb / max(accum.a, 0.00001);
	color_out = vec4(average_color, revealage);
}

