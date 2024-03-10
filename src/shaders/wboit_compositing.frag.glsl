#version 460

// The shader used to composite transparent objects over opaque objects, when using WBOIT/MBOIT.

#ifdef MULTISAMPLED_IMAGE
#define IMAGE_TYPE image2DMS
#define IMAGE_LOAD(image, coord, sample) imageLoad(image, coord, sample)
#else
#define IMAGE_TYPE image2D
#define IMAGE_LOAD(image, coord, sample) imageLoad(image, coord)
#endif

/* sum(rgb * a, a) */
layout(binding = 0, rgba16f) uniform readonly IMAGE_TYPE accum_texture;

/* prod(1 - a) */
layout(binding = 1, r8) uniform readonly IMAGE_TYPE revealage_texture;

layout(location = 0) out vec4 color_out;

void main()
{
	ivec2 load_coord = ivec2(gl_FragCoord.xy);

	vec4 accum = IMAGE_LOAD(accum_texture, load_coord, gl_SampleID);
	float revealage = IMAGE_LOAD(revealage_texture, load_coord, gl_SampleID).r;

	accum = min(accum, 16777216); // suppress overflow

	vec3 average_color = accum.rgb / max(accum.a, 0.00001);
	color_out = vec4(average_color, revealage);
}

