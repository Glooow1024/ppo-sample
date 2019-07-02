#!/bin/bash

#python test.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 \
#    --num-steps 1000 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 \
#    --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 \
#    --num-env-steps 2000000 --use-linear-lr-decay --use-proper-time-limits --gail \
#    --save-dir ./test_models/ --seed 3 --save_expert

SEED=5
SAVE_EXPERT=$true

python test.py --env-name HalfCheetah-v2 --algo ppo --use-gae --log-interval 1 \
    --num-steps 1000 --num-processes 1 --lr 3e-4 --entropy-coef 0 \
    --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 \
    --gamma 0.99 --gae-lambda 0.95 --num-env-steps 3000000 --use-linear-lr-decay \
    --gail \
    --no-cuda --log-dir /tmp/gym/halfcheetah/halfcheetah-0 --seed $SEED \
    --save-dir ./test_models/ --use-proper-time-limits --save_expert $SAVE_EXPERT

python test.py --env-name Walker2d-v2 --algo ppo --use-gae --log-interval 1 \
    --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 \
    --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 \
    --gamma 0.99 --gae-lambda 0.95 --num-env-steps 3000000 --use-linear-lr-decay \
    --no-cuda --log-dir /tmp/gym/walker2d/walker2d-0 --seed $SEED \
    --save-dir ./test_models/ --use-proper-time-limits --save_expert $SAVE_EXPERT
    
python test.py --env-name Hopper-v2 --algo ppo --use-gae --log-interval 1 \
    --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 \
    --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 \
    --gamma 0.99 --gae-lambda 0.95 --num-env-steps 3000000 --use-linear-lr-decay \
    --no-cuda --log-dir /tmp/gym/hopper/hopper-0 --seed $SEED \
    --save-dir ./test_models/ --use-proper-time-limits --save_expert $SAVE_EXPERT
    