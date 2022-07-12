@echo off
echo Note: In case of errors, the file might have non-ASCII unicode characters in it, so make sure the shader source is UTF-8 and *does* have BOM. dxc doesn't seem to handle Unicode characters properly when there is no BOM.
for %%G in (%*) do C:\dxc_2021_12_08\bin\x64\dxc.exe -T vs_6_5 -spirv -fspv-target-env=vulkan1.1 -fvk-use-gl-layout -Fo "../%%~nG.spv"  "%%G"
pause

