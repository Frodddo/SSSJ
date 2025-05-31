CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path /root/DTROcr/SRFormer/SRFormer-Text-Det-main/output_cropped/ \
  --arch ResNet_Scene \
  --decode_type Attention \
  --with_lstm \
  --height 64 \
  --width 256 \
  --max_len 50 \
  --resume //root//DTROcr//RECOG//Text-Recognition-on-Cross-Domain-Datasets-main//model_path//ASTER//model_best.pth.tar \
  --alphabets allcases_symbols \
  --STN_ON \
  --tps_inputsize 32 64 \
  --tps_outputsize 32 100 \
  --tps_margins 0.05 0.05 \
  --stn_activation none \
  --num_control_points 20 \
  