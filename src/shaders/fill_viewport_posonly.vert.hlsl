float4 main(uint vid : SV_VertexID) : SV_POSITION
{
    float2 positions[3] = { { -1.0, -1.0 }, { -1.0, 3.0 }, { 3.0, -1.0 } };
    return float4(positions[min(vid,2)], 0.0, 1.0);
}
