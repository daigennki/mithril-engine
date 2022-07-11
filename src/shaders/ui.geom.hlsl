struct GS_INPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
};
struct GS_OUTPUT
{
    float4 pos : SV_POSITION;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
};
cbuffer material : register(b1)
{
	float4 color;
    float outline;
    float shadow;
};

void offsetTriangle(GS_INPUT input[3], inout TriangleStream<GS_OUTPUT> triStream, float2 offset, float4 color)
{
    GS_OUTPUT output;
    output.color = color;
    for (int i = 0; i < 3; ++i)
    {
        output.pos = input[i].pos;
        output.pos.xy += offset;
        output.uv = input[i].uv;
        triStream.Append(output);
    }
    triStream.RestartStrip();
}

[maxvertexcount(15)]
void main(triangle GS_INPUT input[3], inout TriangleStream<GS_OUTPUT> triStream)
{
    float4 shadowColor = float4(0.0, 0.0, 0.0, 1.0);
    if (outline > 0.0)
    {
        offsetTriangle(input, triStream, float2(outline, outline), shadowColor);
        offsetTriangle(input, triStream, float2(-outline, outline), shadowColor);
        offsetTriangle(input, triStream, float2(-outline, -outline), shadowColor);
        offsetTriangle(input, triStream, float2(outline, -outline), shadowColor);
    }
    else if (shadow > 0.0) offsetTriangle(input, triStream, float2(shadow, shadow), shadowColor);
    offsetTriangle(input, triStream, float2(0.0, 0.0), color);
}