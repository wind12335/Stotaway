#!/bin/bash
# Pack the das vLLM stack for cluster pods whose image lacks it (run on a node that HAS it, e.g. the
# notebook pod). vllm16_vcjob.sh unpacks the tarball into /tmp and puts it on PYTHONPATH when
# `import vllm` fails in the pod. Only torch / numpy are assumed to come with the pod image (11-02: the
# cluster image has NO triton -> triton is packed too; same DTK 26.04 family, preflight fails loudly otherwise).
#
#   bash e2e/dual16/pack_vllm_env.sh [extra top-level names ...]
#
# Contents: vllm + its .libs + dist-info, flash_attn (+ the 2 GB flash_attn_2_cuda .so), the DCU shims
# (amdsmi / lightop / lmslim), and every other site-packages top-level a vLLM run imports that is not
# torch / triton / numpy (list from VS_DUMP_MODS of a 1-card dummy run, see e2e/vllm/vllm_sp.py).
set -euo pipefail
SW_ROOT=${SW_ROOT:-$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)}
SP=$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))")
DEST=${SW_VLLM_TGZ:-$SW_ROOT/3rdparty/pyenv_vllm16.tar.gz}
MODS=${VS_MODS_LIST:-$SW_ROOT/e2e/dual16/vllm_site_closure.txt}
cd "$SP"
LIST="vllm vllm.libs $(ls -d vllm-*.dist-info flash_attn flash_attn-*.dist-info flash_attn_2_cuda*.so amdsmi lightop lmslim 2>/dev/null | tr '\n' ' ')"
if [[ -f "$MODS" ]]; then
    for m in $(grep -vE '^(torch|torchgen|functorch|numpy|nvidia|_distutils_hack|setuptools|pkg_resources)$' "$MODS"); do
        for c in "$m" "$m.py" "$m".*.so "$m.libs"; do [[ -e "$c" ]] && LIST="$LIST $c"; done
    done
fi
# wheel-bundled shared libs (names differ from the import name: pillow.libs for PIL, pyzmq.libs for zmq ...)
LIST="$LIST $(ls -d *.libs | grep -vE '^(torch|numpy)\.libs$' | tr '\n' ' ') $*"
# package metadata (transformers / vllm check versions via importlib.metadata); small, so take all but the base
LIST="$LIST $(ls -d *.dist-info | grep -viE '^(torch|numpy|nvidia)[-_]' | tr '\n' ' ')"
LIST=$(echo $LIST | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo "[pack] from $SP: $LIST"
tar czf "$DEST.tmp" $LIST
N=$(tar tzf "$DEST.tmp" | wc -l)
[[ $N -lt 2000 ]] && { echo "FATAL: only $N entries, refusing"; rm -f "$DEST.tmp"; exit 3; }
mv "$DEST.tmp" "$DEST"
echo "[pack] OK -> $DEST ($N entries, $(du -h "$DEST" | cut -f1))"
