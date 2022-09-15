#!/usr/bin/env bash
#!/bin/sh

# AUDIO

python run_experiments.py --features egemaps --normalize 1 --name sentiment_egemaps --task sentiment --C 5 --weights 4 --gamma scale --poly_degrees 3 --kernel poly  --save_checkpoints

python run_experiments.py --features ds --normalize 1 --name sentiment_ds --task sentiment --C 0.0001 --weights balanced --gamma scale --kernel poly --poly_degrees 4 --num_seeds 5  --save_checkpoints

python run_experiments.py --features wav2vec --normalize 1 --name sentiment_wav2vec --task sentiment --C 10 --weights balanced --gamma auto --kernel rbf --num_seeds 5  --save_checkpoints

# VIDEO

python run_experiments.py --features faus --normalize 1 --name sentiment_faus --task sentiment --C 1 --weights balanced --kernel rbf --gamma scale --num_seeds 5  --save_checkpoints

python run_experiments.py --features vggface2 --normalize 1 --name sentiment_vggface2 --task sentiment --C 0.1 --weights 4 --kernel rbf --poly_degrees 2 --gamma scale --num_seeds 5  --save_checkpoints

python run_experiments.py --features farl --normalize 1 --name sentiment_farl --task sentiment --C 1 --weights balanced --kernel rbf --gamma auto --num_seeds 5  --save_checkpoints

# TEXT

python run_experiments.py --features bert-4-sentence-level --normalize 1 --name sentiment_bert --task sentiment --C 1 --weights 4 --kernel rbf --gamma scale --num_seeds 5  --save_checkpoints

python run_experiments.py --features electra-4-sentence-level --normalize 1 --name sentiment_electra --task sentiment --C 10 --weights 4 --kernel rbf --gamma scale auto --num_seeds 5  --save_checkpoints

python run_experiments.py --features sentiment-bert-4-sentence-level --normalize 1 --name sentiment_sentiment_bert --task sentiment --C 10 --weights balanced --kernel rbf --gamma auto --num_seeds 5  --save_checkpoints