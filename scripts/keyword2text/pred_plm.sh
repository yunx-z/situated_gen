GPU=0
MODEL_NAME=$1
MODEL_NAME=${MODEL_NAME//[\/]/_}
SAVE_DIR=keyword2text/${MODEL_NAME}
# total batch size = 32
PORT=`expr 8888 + ${GPU}`
echo "logs/${SAVE_DIR}_pred.log"

#export CUDA_HOME=/envs/remote/cuda-10.2/
#CUDA_VISIBLE_DEVICES=0,1,2,3 TOKENIZERS_PARALLELISM=false deepspeed \
#    --include localhost:${GPU} \
#    --master_port ${PORT} \
#    methods/keyword2text/plm/run_summarization.py \
#    --deepspeed ds_config_zero2.json \

CUDA_VISIBLE_DEVICES=${GPU} TOKENIZERS_PARALLELISM=false nohup python run_summarization.py \
    --model_name_or_path model_save/${SAVE_DIR} \
    --do_predict \
    --source_prefix "generate two sentences with: " \
    --train_file data/final/keyword2text/train.json \
    --validation_file data/final/keyword2text/dev.json \
    --test_file data/final/keyword2text/test.json \
    --output_dir model_save/${SAVE_DIR} \
    --overwrite_output_dir \
    --per_device_eval_batch_size=32 \
    --max_source_length=128 \
    --max_target_length=256 \
    --text_column keywords \
    --summary_column statement \
    --logging_dir logs/ \
    --report_to none \
    --predict_with_generate \
    > logs/${SAVE_DIR}_pred.log 2>&1 

python methods/keyword2text/coverage.py --test_file data/final/keyword2text/test.json --res_file model_save/keyword2text/${MODEL_NAME}/generated_predictions.txt > logs/${SAVE_DIR}_pred.log 2>&1

python methods/keyword2text/my_eval.py --key_file data/final/keyword2text/test.json --gts_file data/final/keyword2text/test.json --res_file model_save/keyword2text/${MODEL_NAME}/generated_predictions.txt >> logs/${SAVE_DIR}_pred.log 2>&1
