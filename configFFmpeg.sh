# This is a script to configure FFmpeg to compile for MEF.
# Run this script (or just the command below) in the FFmpeg root directory to use it.
# The following configures FFmpeg to compile shared libraries with:
# * VP9 decoder and parser (used for pre-rendered cutscenes)
# * UTVideo decoder (used for RGBA animated textures)
# * Opus audio decoder and parser (used for music and audio accompanying video)
# * PCM Signed 16-bit little-endian decoder (used for lossless audio, just in case)
# * Matroska demuxer (MKV for video,　MKA for audio)
# * DXVA2 VP9 Hardware Acceleration (Windows only, remove on other platforms)
# * VAAPI VP9 Hardware Acceleration (Linux only, remove on other platforms)

./configure --disable-all --enable-shared --enable-avcodec --enable-avformat --enable-swresample --disable-debug --disable-gpl --enable-protocol=file --enable-parser=vp9 --enable-decoder=vp9  --enable-decoder=utvideo --enable-parser=opus --enable-decoder=opus --enable-decoder=pcm_s16le --enable-demuxer=matroska --enable-dxva2 --enable-hwaccel=vp9_dxva2 --enable-vaapi --enable-hwaccel=vp9_vaapi