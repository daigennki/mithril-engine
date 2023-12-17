#version 460
#extension GL_EXT_nonuniform_qualifier: enable

// The shader code used for moment-based OIT moment writes (stage 2).

layout(binding = 0) uniform sampler sampler0;
layout(binding = 1) uniform texture2D textures[];

layout(location = 0) in vec2 texcoord;
layout(location = 1) flat in int instance_index;

layout(location = 0) out vec4 moments;
layout(location = 1) out float optical_depth;
layout(location = 2) out float min_depth;

float depth_to_unit(float z, float c0, float c1)
{
	return log(z * c0) * c1;
}
vec4 make_moments4(float z)
{
	float zsq = z * z;
	return vec4(z, zsq, zsq * z, zsq * zsq);
}
void main()
{
	float alpha = texture(sampler2D(textures[instance_index], sampler0), texcoord).a;
	float depth = gl_FragCoord.z;

	// TODO: use a descriptor set here to reflect changes to the camera near/far planes
	const float near = 0.25;
	const float far = 5000.0;
	const float c0 = 1.0 / near;
	const float c1 = 1.0 / log(far / near);

	const float k_max_alpha = 1.0 - 0.5 / 256.0;

	optical_depth = -log(1.0 - (alpha * k_max_alpha));

	float unit_pos = depth_to_unit(depth, c0, c1);
	moments = make_moments4(unit_pos) * optical_depth;

	min_depth = depth;
}

