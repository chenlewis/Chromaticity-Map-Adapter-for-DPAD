
#d2train d1 test
##python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1test.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_prompt_shallow_test_doc1_Dir"

#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1test.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "128" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_vit_prompt_shallow_Dir"
#d1train d2 test
#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc1traindoc2test.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_prompt_shallow_test_doc2_Dir"

#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc1traindoc2test.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "128" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc2_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_prompt_shallow_test_doc2_Dir"


#d1train d2 test deep
# python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc1traindoc2testdeep.yaml \
#       MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
#       MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "100" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       DATA.DATAPATH "D:\doc2_zl_vit_data" \
#       MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#       OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_29_test\doc1train_prompt_deep_test_doc2_Dir_deep_100"
# sleep 1000
# d2train d1 test deep
python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1testdeep.yaml \
     MODEL.TYPE "vit" DATA.BATCH_SIZE "32"\
     MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "100" \
     DATA.FEATURE "sup_vitb16_imagenet21k" \
     DATA.DATAPATH "D:\doc1_zl_vit_data" \
     MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
     OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_29_test\doc2train_prompt_deep_test_doc1_Dir_deep_100"

sleep 1000
#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1testdeep.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "32" MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "50" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "D:\doc1_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_29_test\doc2train_prompt_deep_test_doc1_Dir_deep"

#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc1traindoc2testdeep.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "32" MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "50" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "D:\doc2_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\5_29_test\doc1train_prompt_deep_test_doc2_Dir_deep"


#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1testdeep.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_prompt_shallow_test_doc1_Dir_deep"
#
#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1testdeep.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_prompt_shallow_test_doc1_Dir_deep"
#
#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1testdeep.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc2_prompt_shallow_test_doc1_Dir_deep"

##python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc2traindoc1test.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "128" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc1_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_vit_prompt_shallow_Dir"
##d1train d2 test deep
#python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc1traindoc2testdeep.yaml \
#      MODEL.TYPE "vit" DATA.BATCH_SIZE "128"\
#      MODEL.PROMPT.DEEP "True" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" \
#      DATA.FEATURE "sup_vitb16_imagenet21k" \
#      DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc2_zl_vit_data" \
#      MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" \
#      OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_prompt_shallow_test_doc2_Dir_deep"
#
##python ../test.py --train-type "prompt" --config-file ../configs/TestPrompt/doc1traindoc2test.yaml MODEL.TYPE "vit" DATA.BATCH_SIZE "128" MODEL.PROMPT.DEEP "False" MODEL.PROMPT.DROPOUT "0.1" MODEL.PROMPT.NUM_TOKENS "10" DATA.FEATURE "sup_vitb16_imagenet21k" DATA.DATAPATH "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\DatatOrigin\doc2_zl_vit_data" MODEL.MODEL_ROOT "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\model\pretrainedModel" OUTPUT_DIR "E:\code\VPTCode\VPT\originCode\vpt-main\vpt-main\OutputDir\doc1_prompt_shallow_test_doc2_Dir"

