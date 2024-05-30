python extract_features.py --model-name=torch://vgg19 \
  --layer-names avgpool \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/13K_round3/ \
  --output-path="output/features_13K_round3_torch_vgg_19_avgpool.pt"
python extract_features.py --model-name=torch://vgg19 \
  --layer-names avgpool \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/dc_test/ \
  --output-path="output/features_dc_test_torch_vgg_19_avgpool.pt"

python extract_features.py --model-name=torch_hub://facebookresearch/dinov2#dinov2_vitg14_reg \
  --layer-names output \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/13K_round3/ \
  --output-path="output/features_13K_round3_torchhub_dinov2_vitg14_reg_output.pt" \
  --batch-size=64
python extract_features.py --model-name=torch_hub://facebookresearch/dinov2#dinov2_vitg14_reg \
  --layer-names output \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/dc_test/ \
  --output-path="output/features_dc_test_torchhub_dinov2_vitg14_reg_output.pt" \
  --batch-size=64

python extract_features.py --model-name=open_clip://ViT-L-14#laion2b_s32b_b82k \
  --layer-names output \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/13K_round3/ \
  --output-path="output/features_13k_round3_open_clip_vit_l_14__laion_features.pt" --batch-size=16
python extract_features.py --model-name=open_clip://ViT-L-14#laion2b_s32b_b82k \
  --layer-names output \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/dc_test/ \
  --output-path="output/features_dc_test_open_clip_vit_l_14__laion_features.pt" --batch-size=16

 python extract_features.py \
  --model-name=open_clip://hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup \
  --layer-names output \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/13K_round3/ \
  --output-path="output/features_13k_round3_open_clip_laion_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup_features.pt" \
  --batch-size=16
 python extract_features.py \
  --model-name=open_clip://hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup \
  --layer-names output \
  --dataset-path=/is/cluster/fast/pmayilvahanan/datasets/dc_test/ \
  --output-path="output/features_dc_test_open_clip_laion_convnext_xxlarge_laion2B_s34B_b82K_augreg_soup_features.pt" \
  --batch-size=16