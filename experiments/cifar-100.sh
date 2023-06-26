#! /bin/bash
# experiment settings
DATASET=cifar-100

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0 1 2 3 4 5 6'
CONFIG_test=configs/test.yaml
CONFIG=configs/cifar-100_prompt.yaml
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR

# Matrix-P

# nohup python3.8 -u run.py --config "$CONFIG_test" --gpuid $GPUID --repeat "$REPEAT" --overwrite "$OVERWRITE" \
#         --learner_type verbose_hint --learner_name ProjHint \
#         --prompt_param 100 10 0.1 \
#         --log_dir ${OUTDIR}/proj_verbose_hint-int-8-newsim-rand10 \
#         >proj_verbose_hint-int-8-newsim-rand10.log 2>&1 &
# nohup python3.8 -u run.py --config "$CONFIG_test" --gpuid $GPUID --repeat "$REPEAT" --overwrite "$OVERWRITE" \
#         --learner_type verbose_hint --learner_name CatHint \
#         --prompt_param 100 10 0.1 \
#         --log_dir ${OUTDIR}/proj_verbose_hint-int-8-newsim-rand10 \
#         >cat_verbose_hint-int-8-newsim-rand10.log 2>&1 &
# nohup python3.8 -u run.py --config "$CONFIG_test" --gpuid $GPUID --repeat "$REPEAT" --overwrite "$OVERWRITE" \
#         --learner_type verbose_hint --learner_name IntHint \
#         --prompt_param 100 10 0.1 \
#         --log_dir ${OUTDIR}/int_verbose_hint-int-8-newsim-rand10 \
#         >int_verbose_hint-int-8-newsim-rand10.log 2>&1 &

# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight/ortho_mu
# nohup python3.8 -u run.py --config "$CONFIG_test" --gpuid $GPUID --repeat "$REPEAT" --overwrite "$OVERWRITE" \
#         --learner_type cls_hint --learner_name MatrixHint \
#         --prompt_param 100 8 0.1 \
#         --log_dir ${OUTDIR}/matrix-test \
#         >matrix-test.log 2>&1 &

# Matrix-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight/ortho_mu
# nohup python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name MatrixPrompt \
#         --prompt_param 100 8 0.1 \
#         --log_dir ${OUTDIR}/matrix-p-8-noAKP \
#         >matrix-p-8-noAKP.log 2>&1 &

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight
# python3.8 -u run.py --config "$CONFIG" --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 300 8 0.1 \
#     --log_dir ${OUTDIR}/coda-p-300
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.1 \
#     --upper_bound_flag \
#     --log_dir ${OUTDIR}/coda-p-upperbound

# DualPrompt
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
nohup python3.8 -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
        --learner_type prompt --learner_name DualPrompt \
        --prompt_param 10 500 50 \
        --log_dir ${OUTDIR}/dual-prompt-500 \
        >dual-prompt-500.log 2>&1 &

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p++
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 1 \
#     --log_dir ${OUTDIR}/l2p++_deep++
