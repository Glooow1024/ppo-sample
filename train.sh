#python main.py --env-name "HalfCheetah-v2" --algo ppo --use-gae --log-interval 1 --num-steps 1000 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits --gail --seed 3

python main.py --env-name Walker2d-v2 --algo ppo --use-gae --log-interval 1 \
    --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 \
    --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 \
    --gamma 0.99 --tau 0.95 --num-env-steps 1000000 --use-linear-lr-decay \
    --no-cuda --log-dir /tmp/gym/walker2d/walker2d-0 --seed 0 \
    --use-proper-time-limits
