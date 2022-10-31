// The shader used to composite transparent objects over opaque objects, when using WBOIT.

/* sum(rgb * a, a) */
Texture2D accum_texture : register(t0);

/* prod(1 - a) */
Texture2D revealage_texture : register(t1);

cbuffer tex_dim : register(b2)
{
	uint2 texture_dimensions;
};

float max_component(float4 color)
{
	return max(max(max(color.r, color.g), color.b), color.a);
}

struct PS_INPUT
{
    float4 position : SV_POSITION;
    float2 texcoords : TEXCOORD;
};

float4 main(PS_INPUT input) : SV_Target
{
    int2 tex_coords_int = int2(input.texcoords * texture_dimensions);
    float revealage = revealage_texture.Load(int3(tex_coords_int, 0)).r;
    if (revealage == 1.0) {
        // Save the blending and color texture fetch cost
        discard;
    }

    float4 accum = accum_texture.Load(int3(tex_coords_int, 0));
    // Suppress overflow
    if (isinf(max_component(abs(accum)))) {
        accum.rgb = float3(accum.a);
    }

    float3 average_color = accum.rgb / max(accum.a, 0.00001);

    // dst' =  (accum.rgb / accum.a) * (1 - revealage) + dst * revealage
    return float4(average_color, 1.0 - revealage);
}

