SamplerState sampler0 : register(s0, space2);
Texture2D base_color : register(t1, space2);

struct PS_INPUT
{
	float4 pos : SV_POSITION;
	float2 uv : TEXCOORD;
	float3 normal : NORMAL;
};

#ifndef TRANSPARENCY_PASS
struct PS_OUTPUT
{
	float4 color : SV_Target0;
};
#endif

/* Environment light */
/*struct DirLight
{
	float3 direction;
	float _filler1;
	float3 color;
	float _filler2;
};
cbuffer directional_light_ubo : register(b0, space3)
{
	DirLight dir_light;
};*/

float3 calc_diff(float3 light_direction, float3 normal, float3 tex_diffuse)
{
	float diff_intensity = max(dot(normal, light_direction), 0.0);
	float3 diffuse = diff_intensity * tex_diffuse;
	return diffuse;
}
// calculate directional light (sunlight/moonlight)
float3 calc_dl(float3 tex_diffuse, float3 normal)
{
	float3 light_direction = normalize(float3(1.0, 1.0, 5.0));
	float3 light_color = float3(12.5, 11.25, 10.0) / 8.0;
	/*float3 light_direction = dir_light.direction;
	float3 light_color = dir_light.color;*/
	float3 ambient = tex_diffuse * 0.04;

	float3 color_out = calc_diff(light_direction, normal, tex_diffuse);
	color_out += ambient;
	return light_color * color_out;
}

// PS_OUTPUT is defined in wboit.hlsl, and will change depending on whether or not TRANSPARENCY_PASS is defined.
PS_OUTPUT main(PS_INPUT input)
{
	float4 tex_color = base_color.Sample(sampler0, input.uv);

#ifdef TRANSPARENCY_PASS
	tex_color.rgb *= tex_color.a;
#endif

	float3 shaded = calc_dl(tex_color.rgb, input.normal);

#ifdef TRANSPARENCY_PASS
	// `write_transparent_pixel` must be defined in the file that defines `TRANSPARENCY_PASS`
	float4 with_alpha = float4(shaded, tex_color.a);
	return write_transparent_pixel(with_alpha, input.pos.z, input.pos.xy);
#else
	PS_OUTPUT output;
	output.color = float4(shaded, 1.0);
	return output;
#endif
}

