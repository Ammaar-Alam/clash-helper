#!/bin/bash
# Fine tune the unit detectors on a Colab GPU VM through the colab CLI (signed in), then fetch the weights.
#
#     research/colab_detector.sh up       # new VM, upload captures/detector and the base weights, install
#     research/colab_detector.sh train    # run research/colab_train.py in the kernel, it keeps the VM alive
#     research/colab_detector.sh log      # follow the training log
#     research/colab_detector.sh fetch    # copy the finished weights into captures/detector/weights
#     research/colab_detector.sh down     # stop the VM
#
# captures/detector comes from research/detector_data.py and holds jobs.json, see colab_train.py.
set -euo pipefail
S=${SESSION:-crdet}
GPU=${GPU:-A100}
REPO=$(cd "$(dirname "$0")/.." && pwd)
DATA=$REPO/captures/detector
RUNS=$HOME/Library/Caches/clash-helper/katacr/KataCR/runs
KEY=$DATA/.ssh_key
[ -f "$KEY" ] || ssh-keygen -q -t ed25519 -N "" -C colab-detector -f "$KEY"
SSH=(ssh -i "$KEY" -o "ProxyCommand=colab ssh --proxy-mode -s $S -i $KEY" -o StrictHostKeyChecking=no
     -o UserKnownHostsFile=/dev/null -o ServerAliveInterval=30 -o ConnectTimeout=120 -o LogLevel=ERROR "root@$S")
PACKAGES="torch==2.2.2 torchvision==0.17.2 ultralytics==8.1.24 numpy==1.26.4 opencv-python-headless==4.9.0.80 lap Pillow scipy
jax==0.4.26 jaxlib==0.4.26 flax==0.8.1 optax==0.2.2 orbax-checkpoint==0.5.3 einops tensorboardX tqdm pandas matplotlib pyyaml"

case "${1:-}" in
up)
  colab new -s "$S" --gpu "$GPU"
  COPYFILE_DISABLE=1 tar -cf - -C "$DATA" --exclude .ssh_key --exclude .ssh_key.pub --exclude weights --exclude build.log . \
    | "${SSH[@]}" 'mkdir -p /content/data /content/out && tar -xf - -C /content/data'
  COPYFILE_DISABLE=1 tar -cf - -C "$RUNS" detector1_v0.7.13.pt detector2_v0.7.13.pt | "${SSH[@]}" 'tar -xf - -C /content/data'
  "${SSH[@]}" 'cat > /content/colab_train.py' < "$REPO/research/colab_train.py"
  # the VM's own Python is too new for the pinned PyTorch, so a 3.11 environment like the Mac's detector
  "${SSH[@]}" "uv venv -q --python 3.11 /content/venv && uv pip install -q --python /content/venv/bin/python $(echo $PACKAGES) \
    > /content/out/install.log 2>&1; tail -3 /content/out/install.log; nvidia-smi --query-gpu=name,memory.total --format=csv"
  ;;
train)
  # a busy kernel keeps the VM, so the training runs inside it and writes its own log
  # the kernel's inline plot backend means nothing to the training's own Python
  printf '%s\n' 'import os, subprocess' \
    'env = dict(os.environ, MPLBACKEND="Agg")' \
    'with open("/content/out/run.log", "a") as log:' \
    '    p = subprocess.Popen(["/content/venv/bin/python", "/content/colab_train.py"], stdout=log, stderr=subprocess.STDOUT, env=env)' \
    '    print("EXIT", p.wait(), flush=True)' > "$DATA/.runner.py"
  nohup colab exec -s "$S" --timeout 86400 -f "$DATA/.runner.py" > "$DATA/.runner.out" 2>&1 &
  echo "training started, follow it with: $0 log"
  ;;
log)
  "${SSH[@]}" "tr '\r' '\n' < /content/out/run.log | grep -v -i warn | tail -${2:-25}"
  ;;
fetch)
  mkdir -p "$DATA/weights"
  "${SSH[@]}" 'cd /content/out && tar -cf - *.pt' | tar -xf - -C "$DATA/weights"
  ls -la "$DATA/weights"
  ;;
down)
  colab stop -s "$S"
  ;;
*)
  sed -n '2,10p' "$0"
  exit 1
  ;;
esac
