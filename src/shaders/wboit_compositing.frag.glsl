#version 450

// The shader used to composite transparent objects over opaque objects, when using WBOIT/MBOIT.

/* sum(rgb * a, a) */
layout(binding = 0, rgba16f) uniform readonly image2D accum_texture;

/* prod(1 - a) */
layout(binding = 1, r8) uniform readonly image2D revealage_texture;

float max_component(vec4 color)
{
	return max(max(max(color.r, color.g), color.b), color.a);
}

layout(location = 0) out vec4 color_out;

void main()
{
	ivec2 load_coord = ivec2(gl_FragCoord.xy);
	float revealage = imageLoad(revealage_texture, load_coord).r;

	if (revealage == 1.0) {
		// Save the blending and color texture fetch cost
		discard;
	}

	vec4 accum = imageLoad(accum_texture, load_coord);
	// Suppress overflow
	if (isinf(max_component(abs(accum)))) {
		accum.rgb = vec3(accum.a);
	}

	vec3 average_color = accum.rgb / max(accum.a, 0.00001);

	color_out = vec4(average_color, revealage);
}

