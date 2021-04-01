@echo off
echo Note: In case of errors, make sure the shader source is plain UTF-8 and does *not* have BOM. glslang doesn't seem to like UTF-8 BOM.
FOR %%G IN (%*) DO glslangValidator -V110 -e "main" --ssb 0 --sub 4 --stb 8 --sbb 8 -o "../%%~nG.spv"  "%%G"
pause