#!/usr/bin/env bash
#!/bin/sh

# AUDIO

python run_rnn_experiments.py --features wav2vec --normalize 0 --name humor_wav2vec --task humor --rnn_hidden_dims 128 --rnn_num_layers 2 --directions bi --lr 0.001 --rnn_dropout 0. --num_seeds 5  --save_checkpoints  --save_predictions

python run_rnn_experiments.py --features egemaps --normalize 1 --name humor_egemaps --task humor --rnn_hidden_dims 64 --rnn_num_layers 2 --directions bi --lr 0.005 --rnn_dropout 0. --num_seeds 5  --save_checkpoints --save_predictions

python run_rnn_experiments.py --features ds --normalize 0 --name humor_ds --task humor --rnn_hidden_dims 64 --rnn_num_layers 4 --directions bi --lr 0.001 --rnn_dropout 0. --num_seeds 5  --save_checkpoints  --save_predictions

# VIDEO

python run_rnn_experiments.py --features farl --normalize 0 --name humor_farl --task humor --rnn_hidden_dims 128 --rnn_num_layers 4 --directions uni --lr 0.0005 --rnn_dropout 0.2 --num_seeds 5 --save_checkpoints --save_predictions

python run_rnn_experiments.py --features faus --normalize 0 --name humor_faus --task humor --rnn_hidden_dims 128 --rnn_num_layers 2 --directions bi --lr 0.00005 --rnn_dropout 0.2 --num_seeds 5 --save_checkpoints --save_predictions

python run_rnn_experiments.py --features vggface2 --normalize 0 --name humor_vggface2 --task humor --rnn_hidden_dims 128 --rnn_num_layers 2 --directions bi --lr 0.0001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions

# TEXT

python run_rnn_experiments.py --features electra --normalize 0 --name humor_electra --task humor --rnn_hidden_dims 128 --rnn_num_layers 4 --directions bi --lr 0.001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions

python run_rnn_experiments.py --features bert --normalize 0 --name humor_bert --task humor --rnn_hidden_dims 64 --rnn_num_layers 4 --directions uni --lr 0.001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions

python run_rnn_experiments.py --features sentiment-bert --normalize 0 --name humor_sentiment_bert --task humor --rnn_hidden_dims 256 --rnn_num_layers 2 --directions uni --lr 0.001 --rnn_dropout 0. --num_seeds 5 --save_checkpoints --save_predictions