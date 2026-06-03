#!/usr/bin/env bash
# Deprecated: use copy_prompthmr_vendor.sh (full rsync, no symlinks).
exec "$(dirname "$0")/copy_prompthmr_vendor.sh" "$@"
