srun --partition=gpu --gres=gpu:tesla:1 --time=60 --mem=64GB --exclude=falcon3 -A revvity --pty /bin/bash
