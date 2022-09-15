#!/usr/bin/env bash
#!/bin/sh

# AUDIO

python run_experiments.py --features egemaps --normalize 1 --name direction_egemaps --task direction --C 0.01 --weights balanced --gamma scale --poly_degrees 3 --kernel poly  --save_checkpoints

python run_experiments.py --features ds --normalize 1 --name direction_ds --task direction --C 1 --weights 4 --gamma scale --kernel rbf --num_seeds 5  --save_checkpoints

python run_experiments.py --features wav2vec --normalize 1 --name direction_wav2vec --task direction --C 0.1 --weights 2 --kernel linear --num_seeds 5  --save_checkpoints

# VIDEO

python run_experiments.py --features faus --normalize 1 --name direction_faus --task direction --C 10 --weights 2 --kernel sigmoid --gamma scale --num_seeds 5  --save_checkpoints

python run_experiments.py --features vggface2 --normalize 1 --name direction_vggface2 --task direction --C 0.1 --weights balanced --kernel poly --poly_degrees 2 --gamma scale --num_seeds 5  --save_checkpoints

python run_experiments.py --features farl --normalize 1 --name direction_farl --task direction --C 2 --weights 8 --kernel rbf --gamma auto --num_seeds 5  --save_checkpoints

# TEXT

python run_experiments.py --features bert --normalize 1 --name direction_bert --task direction --C 0.001 --weights balanced --kernel poly --poly_degrees 4 --gamma scale --num_seeds 5  --save_checkpoints

python run_experiments.py --features electra --normalize 1 --name direction_electra --task direction --C 1 --weights 4 --kernel linear --gamma scale auto --num_seeds 5  --save_checkpoints

python run_experiments.py --features sentiment-bert --normalize 1 --name direction_sentiment_bert --task direction --C 2 --weights balanced --kernel poly --poly_degree 2 --gamma scale --num_seeds 5  --save_checkpoints