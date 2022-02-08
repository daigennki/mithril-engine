echo "Note: In case of errors, make sure the shader source is plain UTF-8 and does *not* have BOM. glslang doesn't seem to like UTF-8 BOM."
for shader in "$@"
do
    glslangValidator -V -e "main" -o "../$(basename $shader .hlsl).spv"  "$shader"
done
