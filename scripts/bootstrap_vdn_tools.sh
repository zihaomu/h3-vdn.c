#!/usr/bin/env bash

set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
tools_dir="$repo_dir/.tools"
python_dir="$tools_dir/python"
ffmpeg_dir="$tools_dir/ffmpeg"
packages_dir="$tools_dir/packages"

mkdir -p "$python_dir" "$ffmpeg_dir" "$packages_dir"

if ! PYTHONPATH="$python_dir" python3 -c 'import huggingface_hub' \
        >/dev/null 2>&1; then
    python3 -m pip install --disable-pip-version-check --target "$python_dir" \
        'huggingface_hub>=1.27,<2'
fi

ffmpeg_packages=(
    ffmpeg libavcodec60 libavdevice60 libavfilter9 libavformat60 libavutil58
    libpostproc57 libswresample4 libswscale7 libjack-jackd2-0 libopenal1
    libsdl2-2.0-0 libpocketsphinx3 libsphinxbase3t64 libbs2b0 liblilv-0-0
    librubberband2 libmysofa1 libflite1 libplacebo338 libass9 libvidstab1.1
    libzimg2 libfftw3-double3 libserd-0-0 libsndio7.0 libsord-0-0
    libsratom-0-0 libunibreak5 libzix-0-0
)

if ! LD_LIBRARY_PATH="$ffmpeg_dir/usr/lib/x86_64-linux-gnu" \
        "$ffmpeg_dir/usr/bin/ffmpeg" -version >/dev/null 2>&1; then
    (
        cd "$packages_dir"
        apt-get download "${ffmpeg_packages[@]}"
        for package in ./*.deb; do
            dpkg-deb -x "$package" "$ffmpeg_dir"
        done
    )
fi

# shellcheck source=use_vdn_tools.sh
. "$script_dir/use_vdn_tools.sh"
python3 -c 'import huggingface_hub; print("huggingface_hub", huggingface_hub.__version__)'
ffmpeg -version | sed -n '1p'
ffprobe -version | sed -n '1p'
