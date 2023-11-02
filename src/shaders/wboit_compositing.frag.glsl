#version 450

// The shader used to composite transparent objects over opaque objects, when using WBOIT/MBOIT.

layout(binding = 0) uniform sampler sampler0;

/* sum(rgb * a, a) */
layout(binding = 1) uniform texture2D accum_texture;

/* prod(1 - a) */
layout(binding = 2) uniform texture2D revealage_texture;

float max_component(vec4 color)
{
	return max(max(max(color.r, color.g), color.b), color.a);
}

layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 color_out;

void main()
{
	float revealage = texture(sampler2D(revealage_texture, sampler0), texcoord).r;

	if (revealage == 1.0) {
		// Save the blending and color texture fetch cost
		discard;
	}

	vec4 accum = texture(sampler2D(accum_texture, sampler0), texcoord);
	// Suppress overflow
	if (isinf(max_component(abs(accum)))) {
		accum.rgb = vec3(accum.a);
	}

	vec3 average_color = accum.rgb / max(accum.a, 0.00001);

	// dst' =  (accum.rgb / accum.a) * (1 - revealage) + dst * revealage
	color_out = vec4(average_color, 1.0 - revealage);
}

