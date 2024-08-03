
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.
#
#python tune_fgvc.py \
#    --train-type "prompt" \
#    --config-file configs/prompt/cub.yaml \
#    MODEL.TYPE "vit" \
#    DATA.BATCH_SIZE "128" \
#    MODEL.PROMPT.DEEP "True" \
#    MODEL.PROMPT.DROPOUT "0.1" \
#    MODEL.PROMPT.NUM_TOKENS "10" \
#    DATA.FEATURE "sup_vitb16_imagenet21k" \
#    DATA.DATAPATH "E:\\code\\研究生工作代码\\VPT\\originCode\\vpt-main\\vpt-main\\DatatOrigin\\CUB_200_2011\\CUB_200_2011" \
#    MODEL.MODEL_ROOT "E:\\code\\研究生工作代码\\VPT\\originCode\\vpt-main\\vpt-main\\model\\pretrainedModel" \
#    OUTPUT_DIR "E:\\code\\研究生工作代码\\VPT\\originCode\\vpt-main\\vpt-main\\OutputDir\\tryDir"
#

%%bash
# launch final training with five random seeds for VTAB-dmlab, sun397 and eurosat. The hyperparameters are the same from our paper.
#model_root=<MODEL_ROOT>
#data_path=<DATA_PATH>
#output_dir=<OUTPUT_DIR
data_path = "E:\\code\\研究生工作代码\\VPT\\originCode\\vpt-main\\vpt-main\\DatatOrigin\\CUB_200_2011\\CUB_200_2011"
model_root = "E:\\code\\研究生工作代码\\VPT\\originCode\\vpt-main\\vpt-main\\model\\pretrainedModel"
output_dir = "E:\\code\\研究生工作代码\\VPT\\originCode\\vpt-main\\vpt-main\\OutputDir\\tryDir"

for seed in "42" "44" "82" "100" "800"; do
    python train.py \
        --config-file configs/prompt/cub.yaml \
        MODEL.TYPE "vit" \
        DATA.BATCH_SIZE "64" \
        MODEL.PROMPT.NUM_TOKENS "100" \
        MODEL.PROMPT.DEEP "True" \
        MODEL.PROMPT.DROPOUT "0.1" \
        DATA.FEATURE "sup_vitb16_imagenet21k" \
        DATA.NAME "vtab-dmlab" \
        DATA.NUMBER_CLASSES "6" \
        SOLVER.BASE_LR "0.25" \
        SOLVER.WEIGHT_DECAY "0.001" \
        SEED ${seed} \
        MODEL.MODEL_ROOT "${model_root}" \
        DATA.DATAPATH "${data_path}" \
        OUTPUT_DIR "${output_dir}/seed${seed}"
done
sleep 10000









