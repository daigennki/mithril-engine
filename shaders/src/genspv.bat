@echo off
echo Note: In case of errors, make sure the shader source is plain UTF-8 and does *not* have BOM. glslang doesn't seem to like UTF-8 BOM.
FOR %%G IN (%*) DO glslangValidator -V -e "main" -o "../%%~nG.spv"  "%%G"
pause
