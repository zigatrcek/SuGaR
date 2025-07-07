#!/bin/bash
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __GL_SYNC_TO_VBLANK=0

xvfb-run -a -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" "$@"
