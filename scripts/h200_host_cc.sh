#!/bin/bash
# Same pinned GCC frontend; explicitly select the native linker for this build.
exec /usr/local/pace-apps/spack/packages/linux-rhel9-x86_64_v3/gcc-11.3.1/gcc-12.3.0-ukkkutsxfl5kpnnaxflpkq2jtliwthfz/bin/gcc -B/usr/bin/ "$@"
