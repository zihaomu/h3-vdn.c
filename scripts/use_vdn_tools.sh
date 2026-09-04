#!/usr/bin/env bash

# Source this file before running h3 or the model download helper. The tools are
# installed under .tools/ and never modify the host Python or system packages.
h3_tools_script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
h3_tools_repo_dir=$(CDPATH= cd -- "$h3_tools_script_dir/.." && pwd)
h3_tools_ffmpeg_root="$h3_tools_repo_dir/.tools/ffmpeg"
h3_tools_python_root="$h3_tools_repo_dir/.tools/python"

if [[ -d $h3_tools_ffmpeg_root/usr/bin ]]; then
    PATH="$h3_tools_ffmpeg_root/usr/bin:$PATH"
    LD_LIBRARY_PATH="$h3_tools_ffmpeg_root/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    H3_FFMPEG="$h3_tools_ffmpeg_root/usr/bin/ffmpeg"
    H3_FFPROBE="$h3_tools_ffmpeg_root/usr/bin/ffprobe"
    export PATH LD_LIBRARY_PATH H3_FFMPEG H3_FFPROBE
fi

if [[ -d $h3_tools_python_root ]]; then
    PYTHONPATH="$h3_tools_python_root${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONPATH
fi

unset h3_tools_script_dir h3_tools_repo_dir
unset h3_tools_ffmpeg_root h3_tools_python_root
