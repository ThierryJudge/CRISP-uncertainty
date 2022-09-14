############################################################################################################################
############################################ CAMUS ####################################################################
############################################################################################################################


####################### TRAIN ##################################################
TRAIN_TAG='TRAIN_LUNG'
#python runner.py +exp=lung/train-lung-baseline   comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.1
#python runner.py +exp=lung/train-lung-mc         comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.5
#python runner.py +exp=lung/train-lung-mc         comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.25
#python runner.py +exp=lung/train-lung-lce        comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=lung/train-lung-confidnet  comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=lung/train-lung-crisp       comet_tags=[${TRAIN_TAG}]
#
#python runner.py +exp=lung/lung-crisp-samples     comet_tags=[${TRAIN_TAG}]


###################### TEST ##################################################conf
#TEST_TAG='TEST-LUNG-MONT'
#python runner.py +exp=lung/test-lung-entropy    comet_tags=[${TEST_TAG}]
#python runner.py +exp=lung/test-lung-mc         comet_tags=[${TEST_TAG}]  system.module.dropout=0.5
#python runner.py +exp=lung/test-lung-mc         comet_tags=[${TEST_TAG}]  system.module.dropout=0.25
#python runner.py +exp=lung/test-lung-lce        comet_tags=[${TEST_TAG}]
#python runner.py +exp=lung/test-lung-morph      comet_tags=[${TEST_TAG}]
#python runner.py +exp=lung/test-lung-confidnet  comet_tags=[${TEST_TAG}]
#python runner.py +exp=lung/test-lung-crisp       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-segmentation-DA\${data.da}-\${seed}.ckpt
#python runner.py +exp=lung/test-lung-crisp       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-mcdropout25-DA\${data.da}-\${seed}.ckpt   system.iterations=10
#python runner.py +exp=lung/test-lung-crisp       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-mcdropout50-DA\${data.da}-\${seed}.ckpt   system.iterations=10
#python runner.py +exp=lung/test-lung-crisp       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-learnconf-DA\${data.da}-\${seed}.ckpt


TEST_TAG='TEST-LUNG-JSTR'
python runner.py +exp=lung/test-lung-entropy    data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}]
python runner.py +exp=lung/test-lung-mc         data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}]  system.module.dropout=0.5
python runner.py +exp=lung/test-lung-mc         data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}]  system.module.dropout=0.25
python runner.py +exp=lung/test-lung-lce        data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}]
python runner.py +exp=lung/test-lung-morph      data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}]
python runner.py +exp=lung/test-lung-confidnet  data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}]
python runner.py +exp=lung/test-lung-clip       data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-segmentation-DA\${data.da}-\${seed}.ckpt
python runner.py +exp=lung/test-lung-clip       data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-mcdropout25-DA\${data.da}-\${seed}.ckpt   system.iterations=10
python runner.py +exp=lung/test-lung-clip       data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-mcdropout50-DA\${data.da}-\${seed}.ckpt   system.iterations=10
python runner.py +exp=lung/test-lung-clip       data.dataset_path=\${oc.env:JSTR_DATA_PATH} comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/lung-learnconf-DA\${data.da}-\${seed}.ckpt
