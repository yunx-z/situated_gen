GPU=$1
#CORPUSS=('QA_DATASETS' 'WIKI' 'SIMPLEWIKI')
#CORPUSS=($2)
CORPUSS=('QA_DATASETS')

for CORPUS in ${CORPUSS[@]}
do
	LOG_FILE="logs/data_collection/build_${CORPUS}.log"
	echo "check out ${LOG_FILE}"
	CUDA_VISIBLE_DEVICES=${GPU} nohup python data_collection/build_dataset.py --corpus ${CORPUS} > ${LOG_FILE} 2>&1 &
done
