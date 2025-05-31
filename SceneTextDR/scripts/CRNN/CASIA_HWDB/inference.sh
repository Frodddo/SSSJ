CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path /root/DTROcr/SRFormer/SRFormer-Text-Det-main/output_cropped/ \
  --arch ResNet_IAM \
  --decode_type CTC \
  --with_lstm \
  --height 192 \
  --width 2048 \
  --max_len 128 \
  --resume //root//DTROcr//RECOG//Text-Recognition-on-Cross-Domain-Datasets-main//model_path//crnn//handw//model_best.pth.tar \
  --alphabets casia_360cc \
  --padresize \
  