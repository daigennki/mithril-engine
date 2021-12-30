# Run one of the commands below in the FFmpeg root directory to configure FFmpeg to compile for MEF.
# The following configures FFmpeg to compile shared libraries with:
# * VP9 decoder and parser (used for pre-rendered cutscenes)
# * UTVideo decoder (used for RGBA animated textures)
# * Opus audio decoder and parser (used for music and audio accompanying video)
# * PCM Signed 16-bit little-endian decoder (used for lossless audio, just in case)
# * Matroska demuxer (MKV for video,　MKA for audio)
# * VP9 Hardware acceleration:
#   * Windows: DXVA2
#   * Linux: VAAPI

# Linux:
./configure --disable-all --enable-shared --enable-avcodec --enable-avformat --enable-swresample --disable-debug --disable-gpl --enable-protocol=file --enable-parser=vp9 --enable-decoder=vp9  --enable-decoder=utvideo --enable-parser=opus --enable-decoder=opus --enable-decoder=pcm_s16le --enable-demuxer=matroska --enable-vaapi --enable-hwaccel=vp9_vaapi --prefix=./built

# Windows (cross-compile from Linux or WSL):
./configure --disable-all --enable-shared --enable-avcodec --enable-avformat --enable-swresample --disable-debug --disable-gpl --enable-protocol=file --enable-parser=vp9 --enable-decoder=vp9  --enable-decoder=utvideo --enable-parser=opus --enable-decoder=opus --enable-decoder=pcm_s16le --enable-demuxer=matroska --enable-dxva2 --enable-hwaccel=vp9_dxva2 --target-os=mingw64 --arch=x86_64 --cross-prefix=x86_64-w64-mingw32- --prefix=./built