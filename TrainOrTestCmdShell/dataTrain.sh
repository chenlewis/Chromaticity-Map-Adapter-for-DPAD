
#d2 shallow
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_vit_prompt_shallow_Dir2"

#
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml  MODEL.TYPE "vit" DATA.BATCH_SIZE "64" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "D:\doc2_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel"  OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_vit_prompt_shallow_Dir2"

#d2 deep
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "64"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.2" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_vit_prompt_deep_Dir2"

#d2 shallow
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "100" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_vit_prompt_shallow_Dir2_len50_64"

#d2 shallow 100
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "200" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_vit_prompt_shallow_Dir2_len100_64"

#d2 shallow 200
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "200" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc2_vit_prompt_shallow_Dir2_len200_32"

#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "32" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "200" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "D:\doc2_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc2_vit_prompt_shallow_Dir2_len200_32"



#d2 deep 200
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "200" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc2_vit_prompt_deep_Dir2_len200_32"


###d1 shallow
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_vit_prompt_shallow_Dir2"


#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "128" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_vit_prompt_shallow_Dir"

# d1 deep
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_vit_prompt_deep_Dir2"

##d1 shallow
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "50" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_vit_prompt_shallow_Dir2_len50"

###d1 shallow 100
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "100" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_vit_prompt_shallow_Dir2_len100"


##d1 shallow 200
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "200" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc1_vit_prompt_shallow_Dir2_reallen200_32_nophone2"


#
##python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "128" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_vit_prompt_shallow_Dir"
#
# d1 deep
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "50" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc1_zl_vit_data"  \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc1_vit_prompt_deep_Dir2_len50_batchsize32_1_8"
#
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "100" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc1_zl_vit_data"  \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc1_vit_prompt_deep_Dir2_len100_batchsize32_1_8"
1
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc1.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "200" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc1_zl_vit_data"  \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc1_vit_prompt_deep_Dir2_len200_batchsize32_1_8_adam"

#d2 deep
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "50" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc2_vit_prompt_deep_Dir2_len50_batchsize32_1_8"

python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/S1.yaml \
      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "100" \
      DATA.FEATURE "sup_vitb16_imagenet21k" \
      DATA.DATAPATH "/data/data5/llw_data/vpt_data/S1" \
      MODEL.MODEL_ROOT "/data/data3/llw/coding/vpt/pretrainModel" \
      OUTPUT_DIR "/data/data3/llw/coding/vpt/output/S1"
#
#python ../tune_fgvc.py --train-type "prompt" --config-file ../configs/prompt/doc2.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "200" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "D:\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_28\doc2_vit_prompt_deep_Dir2_len200_batchsize32"