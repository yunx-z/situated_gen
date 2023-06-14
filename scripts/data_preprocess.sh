GPU=$1
#DATASETS=('creak' 'strategyqa' 'commonsenseqa')
DATASETS=($2)

for DATASET in ${DATASETS[@]}
do
	LOG_FILE="logs/data_collection/${DATASET}.log"
	echo "check out ${LOG_FILE}"
	CUDA_VISIBLE_DEVICES=${GPU} nohup python data_collection/preprocess.py --dataset ${DATASET} > ${LOG_FILE} 2>&1 &
done
