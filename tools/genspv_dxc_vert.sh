echo "Note: In case of errors, the file might have non-ASCII unicode characters in it, so make sure the shader source is UTF-8 and *does* have BOM. dxc doesn't seem to handle Unicode characters properly when there is no BOM."
for shader in "$@"
do
    /opt/directx-shader-compiler/bin/dxc -T vs_6_5 -spirv -fspv-target-env=vulkan1.1 -fvk-use-gl-layout -fvk-b-shift 4 0 -fvk-b-shift 4 1 -fvk-b-shift 4 2 -Fo "../$(basename $shader .hlsl).spv"  "$shader"
done