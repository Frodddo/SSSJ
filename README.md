# 随时随景，随时随译v2项目使用指北

## 训练模型权重下载与配置

百度网盘：https://pan.baidu.com/s/1sWLb9CckBjo_b249Ksx9gg 提取码：0531

1. openocr_repsvtr_ch.pth放在./OpenOCR下

2. Pretrain_ArT.pth totaltext-srformer-3seg.pth art-srformer-1seg.pth ctw1500-srformer-3seg.pth放在./SceneTextDR/model_weights下
3. model_best.pth.tar放在./SceneTextDR/model_path/ASTER下
4. model_best2.pth.tar放在./SceneTextDR/model_path/crnn/handw下



## 工程简介

1. 本仓库为服务端代码
2. ./Server.java为与客户端建立socket通信的程序
3. 主文件为./SceneTextDR/main.py



## 项目简介

本项目旨在完成自然场景下的文字检测、识别、翻译一体化的任务，并使用大语言模型辅助识别翻译结果

1. 文字检测网络：SRFormer

   论文：https://arxiv.org/pdf/2308.10531v2

   官方repo：https://github.com/retsuh-bqw/SRFormer-Text-Det 

2. 文字识别网络：CRNN&OPENOCR

3. 文字翻译API：DEEPL

4. 大语言模型：DEEPSEEK R1/V3



