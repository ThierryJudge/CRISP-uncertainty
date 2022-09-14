############################################################################################################################
############################################ HMC_QU ####################################################################
############################################################################################################################


####################### TRAIN ##################################################
#TRAIN_TAG='TRAIN-CAMUS-LV'
#python runner.py +exp=train-camus-lce        data.labels=[BG,lv]   data.da='pixel' comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=train-camus-baseline   data.labels=[BG,lv]   data.da='pixel' comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.1
#python runner.py +exp=train-camus-mc         data.labels=[BG,lv]   data.da='pixel' comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.5
#python runner.py +exp=train-camus-mc         data.labels=[BG,lv]   data.da='pixel' comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.25
#python runner.py +exp=train-camus-confidnet  data.labels=[BG,lv]   data.da='pixel' comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=train-camus-crisp       data.labels=[BG,lv]   data.da='pixel' comet_tags=[${TRAIN_TAG}]

### Generate latent samples from CAMUS dataset ###
python runner.py +exp=lv-clip-samples      data.labels=[BG,lv]                  comet_tags=[${TRAIN_TAG}]


####################### TEST ##################################################conf
#TEST_TAG='TEST-HMCQU'
##python runner.py +exp=test-hmcqu-entropy    comet_tags=[${TEST_TAG}]
##python runner.py +exp=test-hmcqu-mc         comet_tags=[${TEST_TAG}]  system.module.dropout=0.5
##python runner.py +exp=test-hmcqu-mc         comet_tags=[${TEST_TAG}]  system.module.dropout=0.25
##python runner.py +exp=test-hmcqu-lce        comet_tags=[${TEST_TAG}]
##python runner.py +exp=test-hmcqu-confidnet  comet_tags=[${TEST_TAG}]
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da=null system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-segmentation-\${seed}.ckpt
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da=null system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-mcdropout25-\${seed}.ckpt   system.iterations=10
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da=null system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-mcdropout50-\${seed}.ckpt   system.iterations=10
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da=null system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-learnconf-\${seednull
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da='pixel' system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-segmentation-\${seed}.ckpt
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da='pixel' system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-mcdropout25-\${seed}.ckpt   system.iterations=10
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da='pixel' system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-mcdropout50-\${seed}.ckpt   system.iterations=10
#python runner.py +exp=test-hmcqu-crisp       comet_tags=[${TEST_TAG}] +data.da='pixel' system.module_ckpt=\${oc.env:SAVE_PATH}/camus-MYO-learnconf-\${seed}.ckpt


