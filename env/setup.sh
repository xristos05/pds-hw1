#!/usr/bin/env bash
# Install the tools for PDS Homework 1.
#
#   bash env/setup.sh
#
# Ubuntu 24.04 (Codespaces, WSL, Lima) on x86_64; on other systems install
# a C compiler with OpenMP, OpenCilk (https://www.opencilk.org), and matio
# with HDF5 support yourself.
# Safe to run more than once. Its output is saved in ~/hw1-setup.log.

set -u
LOG="$HOME/hw1-setup.log"
exec > >(tee "$LOG") 2>&1
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*"; echo "Ask on Zulip and attach $LOG"; exit 1; }

cd "$(dirname "$0")/.." || die "cannot find the repository"
[ "$(id -u)" -ne 0 ] || die "Do not run this script as root. Run: bash env/setup.sh"
command -v apt-get >/dev/null || die "Not an Ubuntu/Debian system: see the comments at the top of this script."

step "Compilers and libraries (gcc, make, matio, HDF5 tools)"
sudo apt-get update -qq || die "apt-get update failed"
sudo apt-get install -y -qq build-essential libmatio-dev hdf5-tools unzip curl \
  || die "apt-get install failed"

OPENCILK=3.0.0
DIR=/opt/opencilk-$OPENCILK
if [ ! -x "$DIR/bin/clang" ]; then
  step "OpenCilk $OPENCILK"
  [ "$(uname -m)" = x86_64 ] || die "The OpenCilk release here is for x86_64; see https://www.opencilk.org/doc/users-guide/install"
  T=opencilk-$OPENCILK-x86_64-linux-gnu-ubuntu-24.04.tar.gz
  curl -fL -o /tmp/$T "https://github.com/OpenCilk/opencilk-project/releases/download/opencilk/v3.0/$T" \
    || die "OpenCilk download failed"
  sudo mkdir -p "$DIR" && sudo tar -xzf /tmp/$T -C "$DIR" --strip-components=1 \
    || die "OpenCilk unpack failed"
  rm -f /tmp/$T
fi
grep -q "$DIR/bin" "$HOME/.bashrc" 2>/dev/null || echo "export PATH=\"$DIR/bin:\$PATH\"" >> "$HOME/.bashrc"
export PATH="$DIR/bin:$PATH"

[ -f data/k49_norm_10NN.mat ] || die "data/ is missing: clone the whole repository"

step "Check: build and run the starter"
make -C starter -s || die "the starter does not build"
./starter/readdata data/k49_norm_10NN || die "the starter cannot read the data"
clang --version | head -1

printf '\n\033[1;32mReady.\033[0m Open a new terminal (or run: source ~/.bashrc) to use OpenCilk'"'"'s clang.\n'
