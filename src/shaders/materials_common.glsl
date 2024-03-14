#extension GL_EXT_nonuniform_qualifier: enable

// Specialization constant indicating if this fragment shader is for a transparency (OIT) pass.
layout(constant_id = 0) const bool TRANSPARENCY_PASS = false;

/* Material parameters */
layout(binding = 0) uniform sampler sampler0;

// All textures will be taken from here, with textures for different purposes being at different
// offsets. Add the offset to the instance index so that the correct texture for the current
// material gets selected.
layout(binding = 1) uniform texture2D textures[];

/* Lighting stuff */
#define CSM_COUNT 3

layout(set = 1, binding = 0) uniform samplerShadow shadow_sampler;
struct DirLight
{
	mat4 projviews[CSM_COUNT];
	vec3 direction;
	float _filler1;
	vec4 color_intensity;
};
layout(set = 1, binding = 1) uniform dir_light_ubo
{
	DirLight dir_light;
};
layout(set = 1, binding = 2) uniform texture2DArray dir_light_shadow;

/* Shader outputs */
layout(location = 0) out vec4 color_out;  // used as color output in opaque pass, and accum in transparency pass
layout(location = 1) out float revealage; // unused in opaque pass; only used in transparency pass

// Use this function in your material's fragment shader to write pixels to the output image.
void write_pixel(vec4 shaded_with_alpha)
{
	if (TRANSPARENCY_PASS) {
		// Implementation of equation 10 (one of the "generic weight functions") in the WBOIT paper.
		float w = shaded_with_alpha.a * max(1e-2, 3e3 * pow(1.0 - gl_FragCoord.z, 3));
		color_out = shaded_with_alpha * w;
		revealage = shaded_with_alpha.a;
	} else {
		color_out = vec4(shaded_with_alpha.rgb, 1.0);
	}
}
