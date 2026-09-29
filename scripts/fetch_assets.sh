#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
commit=c96a32d28fb5da84da38c1da4d749e7a13212855
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git clone --quiet --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git "$work/repo"
git -C "$work/repo" sparse-checkout set universal_robots_ur5e robotiq_2f85
git -C "$work/repo" checkout --quiet "$commit"
mkdir -p "$root/assets/vendor/menagerie"
for name in universal_robots_ur5e robotiq_2f85; do
  rm -rf "$root/assets/vendor/menagerie/$name"
  cp -a "$work/repo/$name" "$root/assets/vendor/menagerie/$name"
done
