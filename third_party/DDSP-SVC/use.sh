#预处理
python preprocess.py -c configs/reflow.yaml -j 4

#训练
python train_reflow.py -c configs/reflow.yaml

#推理
python main_reflow.py \
    -i "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/0003/cleaned/step_2/Instrumental/ヨルシカ - だから僕は音楽を辞めた_Instrumental.flac" \
    -m "exp/reflow-test/model_40000.pt" \
    -o "/home/pangjichen/workspace/VocalTS/linked_data/inference_data/0003/transfered/ヨルシカ - だから僕は音楽を辞めた_Harmonic.flac" \
    -k 0 \
    -id 1 \
    -method "auto" \
    -ts 0.0