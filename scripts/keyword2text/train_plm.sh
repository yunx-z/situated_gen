GPU=$1
MODEL_NAME=$2
SAVE_DIR=keyword2text/${MODEL_NAME//[\/]/_}
# total batch size = 32
PORT=`expr 8888 + ${GPU}`
echo "logs/${SAVE_DIR}.log"

#export CUDA_HOME=/usr/local/cuda
#CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false deepspeed \
#    --include localhost:${GPU} \
#    --master_port ${PORT} \
#    methods/keyword2text/plm/run_summarization.py \
#    --deepspeed ds_config_zero2.json \

CUDA_VISIBLE_DEVICES=${GPU} TOKENIZERS_PARALLELISM=false nohup python run_summarization.py \
    --model_name_or_path ${MODEL_NAME} \
    --do_train \
    --num_train_epochs  10.0  \
    --do_eval \
    --do_predict \
    --evaluation_strategy epoch \
    --source_prefix "generate two sentences with: " \
    --train_file data/final/keyword2text/train.json \
    --validation_file data/final/keyword2text/dev.json \
    --test_file data/final/keyword2text/test.json \
    --output_dir model_save/${SAVE_DIR} \
    --overwrite_output_dir \
    --per_device_train_batch_size=1 \
    --per_device_eval_batch_size=1 \
    --max_source_length=64 \
    --max_target_length=128 \
    --gradient_accumulation_steps 32 \
    --text_column keywords \
    --summary_column statement \
    --logging_dir logs/runs/${SAVE_DIR} \
    --save_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model "eval_rouge2" \
    --learning_rate 3e-5 \
    --warmup_steps 500 \
    --report_to tensorboard \
    --predict_with_generate \
    > logs/${SAVE_DIR}.log 2>&1 &

#rm -r model_save/${SAVE_DIR}/checkpoint-*
#rm -r model_save/${SAVE_DIR}/global_step*
#    --max_train_samples 100 \
#    --max_eval_samples 100 \
#    --max_predict_samples 100 \
#    --max_steps 10 \

