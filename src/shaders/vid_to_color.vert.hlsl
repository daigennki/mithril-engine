cbuffer model : register(b0)
{
	float4x4 transform;	    // rotation, scale, and translation
};
cbuffer projviewmat : register(b0, space1)
{
    float4x4 projview;
};
struct VS_INPUT
{
    float3 position : POSITION;
    float2 uv : TEXCOORD;
};
struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float3 color : COLOR;
};
VS_OUTPUT main(VS_INPUT input, uint vid : SV_VertexID)
{
    VS_OUTPUT output;
    output.pos = mul(transform, float4(input.position, 1.0));
    output.pos = mul(projview, output.pos);
    	
	// multiply vertex ID to get different hues, with constant saturation and variation
	float3 hsv = float3(vid * 120.0 / 3.6, 100.0, 100.0) * 0.01;

	// then convert HSV to RGB
	// source: https://web.archive.org/web/20200207113336/http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl
	float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
	float3 p = abs(frac(hsv.xxx + K.xyz) * 6.0 - K.www);
	output.color = hsv.z * lerp(K.xxx, saturate(p - K.xxx), hsv.y);

    return output;
}
