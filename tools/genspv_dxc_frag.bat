@echo off
echo Note: In case of errors, the file might have non-ASCII unicode characters in it, so make sure the shader source is UTF-8 and *does* have BOM. dxc doesn't seem to handle Unicode characters properly when there is no BOM.
for %%G in (%*) do C:/dxc-artifacts/bin/dxc -T ps_6_5 -spirv -fspv-target-env=vulkan1.1 -fvk-use-gl-layout -fvk-b-shift 4 0 -fvk-t-shift 8 0 -fvk-b-shift 4 1 -fvk-t-shift 8 1 -fvk-b-shift 4 2 -fvk-t-shift 8 2 -Fo "../%%~nG.spv"  "%%G"
pause