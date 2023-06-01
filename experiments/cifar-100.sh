#! /bin/bash
# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0 1 2 3'
CONFIG=configs/cifar-100_prompt.yaml
CONFIG_test=configs/test.yaml
REPEAT=1
OVERWRITE=1

###############################################################

# process inputs
mkdir -p $OUTDIR

# Matrix-P

# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight/ortho_mu
# nohup python -u run.py --config "$CONFIG_test" --gpuid $GPUID --repeat "$REPEAT" --overwrite "$OVERWRITE" \
#         --learner_type prompt --learner_name MatrixPrompt \
#         --prompt_param 100 8 0.1 \
#         --log_dir ${OUTDIR}/test-8-freeze_linear_cosine \
#         >test-8-freeze_linear_cosine.log 2>&1 &
nohup python -u run.py --config "$CONFIG_test" --gpuid $GPUID --repeat "$REPEAT" --overwrite "$OVERWRITE" \
        --learner_type prompt --learner_name IntPrompt \
        --prompt_param 100 20 0.1 \
        --log_dir ${OUTDIR}/int-8-rndchs_07_mat \
        >int-8-rndchs_07_mat.log 2>&1 &


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
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name CODAPrompt \
#     --prompt_param 100 8 0.1 \
#     --log_dir ${OUTDIR}/coda-p
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
# nohup python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#         --learner_type prompt --learner_name DualPrompt \
#         --prompt_param 10 160 160 \
#         --log_dir ${OUTDIR}/dual-prompt-160-miniloss \
#         >dual-prompt-p-8-soft.log 2>&1 &

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
