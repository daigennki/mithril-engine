#version 460

// The shader used to composite transparent objects over opaque objects, when using WBOIT/MBOIT.

/* sum(rgb * a, a) */
layout(binding = 0, rgba16f) uniform readonly image2DMS accum_texture;

/* prod(1 - a) */
layout(binding = 1, r8) uniform readonly image2DMS revealage_texture;

layout(location = 0) out vec4 color_out;

void main()
{
	ivec2 load_coord = ivec2(gl_FragCoord.xy);

	vec4 accum = imageLoad(accum_texture, load_coord, gl_SampleID);
	accum = min(accum, 16777216); // suppress overflow

	float revealage = imageLoad(revealage_texture, load_coord, gl_SampleID).r;

	vec3 average_color = accum.rgb / max(accum.a, 0.00001);
	color_out = vec4(average_color, revealage);
}

