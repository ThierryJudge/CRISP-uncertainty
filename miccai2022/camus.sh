############################################################################################################################
############################################ CAMUS ####################################################################
############################################################################################################################


####################### TRAIN ##################################################
#TRAIN_TAG='TRAIN-CAMUS-LVMYO'
#python runner.py +exp=train-camus-baseline   data.labels=[BG,LV,MYO] comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.1
#python runner.py +exp=train-camus-mc         data.labels=[BG,LV,MYO] comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.5
#python runner.py +exp=train-camus-mc         data.labels=[BG,LV,MYO] comet_tags=[${TRAIN_TAG}]  system.module.dropout=0.25
#python runner.py +exp=train-camus-lce        data.labels=[BG,LV,MYO] comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=train-camus-confidnet  data.labels=[BG,LV,MYO] comet_tags=[${TRAIN_TAG}]
#python runner.py +exp=train-camus-crisp       data.labels=[BG,LV,MYO] comet_tags=[${TRAIN_TAG}]


###################### TEST ##################################################conf
TEST_TAG='TEST-LVMYO-DA-6'
#python runner.py +exp=test-camus-entropy    data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}]
#python runner.py +exp=test-camus-mc         data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}]  system.module.dropout=0.5
#python runner.py +exp=test-camus-mc         data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}]  system.module.dropout=0.25
#python runner.py +exp=test-camus-lce        data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}]
#python runner.py +exp=test-camus-morph      data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}]
#python runner.py +exp=test-camus-confidnet  data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}]
python runner.py +exp=test-camus-crisp       data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-LV-MYO-segmentation-\${seed}.ckpt
python runner.py +exp=test-camus-crisp       data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-LV-MYO-mcdropout25-\${seed}.ckpt   system.iterations=10
python runner.py +exp=test-camus-crisp       data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-LV-MYO-mcdropout50-\${seed}.ckpt   system.iterations=10
python runner.py +exp=test-camus-crisp       data.labels=[BG,LV,MYO] data.test_da=True comet_tags=[${TEST_TAG}] system.module_ckpt=\${model_path}/camus-LV-MYO-learnconf-\${seed}.ckpt
