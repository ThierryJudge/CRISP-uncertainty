############################################################################################################################
############################################ HMC_QU ####################################################################
############################################################################################################################


####################### TRAIN ##################################################
#TRAIN_TAG='TRAIN-CAMUS-MYO'
#python runner.py +exp=hmc-qu/train-camus-lce        data.labels=[BG,MYO]   data.da='pixel' comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=hmc-qu/train-camus-baseline   data.labels=[BG,MYO]   data.da='pixel' comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.1
#python runner.py +exp=hmc-qu/train-camus-mc         data.labels=[BG,MYO]   data.da='pixel' comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.5
#python runner.py +exp=hmc-qu/train-camus-mc         data.labels=[BG,MYO]   data.da='pixel' comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.25
#python runner.py +exp=hmc-qu/train-camus-confidnet  data.labels=[BG,MYO]   data.da='pixel' comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=hmc-qu/train-camus-crisp       data.labels=[BG,MYO]   data.da='pixel' comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=hmc-qu/hmcqu-crisp-samples     data.labels=[BG,MYO]  comet_tags=[${TRAIN_TAG}] # Generate latent samples from CAMUS dataset ###


####################### TEST ##################################################conf
TEST_TAG='TEST-HMCQU-FINAL'
python runner.py +exp=hmc-qu/test-hmcqu-entropy    comet_tags=[${TEST_TAG}]
python runner.py +exp=hmc-qu/test-hmcqu-morph      comet_tags=[${TEST_TAG}]
python runner.py +exp=hmc-qu/test-hmcqu-mc         comet_tags=[${TEST_TAG}]  system.module.dropout=0.5
python runner.py +exp=hmc-qu/test-hmcqu-mc         comet_tags=[${TEST_TAG}]  system.module.dropout=0.25
python runner.py +exp=hmc-qu/test-hmcqu-lce        comet_tags=[${TEST_TAG}]
python runner.py +exp=hmc-qu/test-hmcqu-confidnet  comet_tags=[${TEST_TAG}]
python runner.py +exp=hmc-qu/test-hmcqu-clip       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-MYO-segmentation-\${seed}.ckpt system.decode=false
python runner.py +exp=hmc-qu/test-hmcqu-clip       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-MYO-mcdropout25-\${seed}.ckpt   system.iterations=10
python runner.py +exp=hmc-qu/test-hmcqu-clip       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-MYO-mcdropout50-\${seed}.ckpt   system.iterations=10
python runner.py +exp=hmc-qu/test-hmcqu-clip       comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-MYO-learnconf-\${seed}.ckpt


