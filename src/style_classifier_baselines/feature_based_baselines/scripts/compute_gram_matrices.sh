python compute_gram_matrices.py --model-name=torch://vgg19 \
  --layer-names features_0 features_5 features_10 features_19 features_28 \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/13K_round3/ \
  --output-path="output/gram_features_13K_round3_torch_vgg_19_features_0_5_10_19_28.pt"
python compute_gram_matrices.py --model-name=torch://vgg19 \
  --layer-names features_0 features_5 features_10 features_19 features_28 \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/dc_test/ \
  --output-path="output/gram_features_dc_test_torch_vgg_19_features_0_5_10_19_28.pt"