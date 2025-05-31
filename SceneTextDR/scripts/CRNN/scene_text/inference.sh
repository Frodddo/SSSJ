CUDA_VISIBLE_DEVICES=0,1 python inferrence.py \
  --image_path test_images/1.png \
  --arch ResNet_Scene \
  --decode_type CTC \
  --with_lstm \
  --height 32 \
  --width 100 \
  --max_len 25 \
  --lower \
  # --resume runs/best_model/CRNN/scene/model_best.pth.tar \
  --resume D:\\大创\\新的开始\\Text-Recognition-on-Cross-Domain-Datasets-main\\model_path\\ASTER\\model_best.pth.tar \
  --alphabets lowercase\
