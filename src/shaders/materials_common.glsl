//#include "mboit_weights.glsl"
#include "wboit_accum.glsl"

// Specialization constant to specify if this is for a transparency (OIT) pass.
layout(constant_id = 0) const bool TRANSPARENCY_PASS = false;

/* Material parameters */
layout(binding = 0) uniform sampler sampler0;

// All textures will be taken from here, with textures for different purposes being at different offsets.
// The offsets will be added to the instance index so that the correct texture for the current material gets selected.
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

// The shader output is defined in wboit_accum.glsl.

// Use this function in your material's fragment shader to write pixels to the output image.
void write_pixel(vec4 shaded_with_alpha)
{
	if (TRANSPARENCY_PASS) {
		write_transparent_pixel(shaded_with_alpha);
	} else {
		color_out = vec4(shaded_with_alpha.rgb, 1.0);
	}
}
