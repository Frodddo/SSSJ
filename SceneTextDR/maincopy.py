import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import numpy as np

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from demo.predictor import VisualizationDemo
from adet.config import get_cfg
import multiprocessing as mp
import gc

import torch
import PIL.Image as Image
import torchvision
import time
import sys
import os
import torch.nn.functional as F

from torchvision import transforms
from config import get_args
from lib.models.model_builder_CTC import ModelBuilder_CTC
from lib.models.model_builder_Attention import ModelBuilder_Att
from lib.models.model_builder_DAN import ModelBuilder_DAN
from lib.utils.labelmaps import CTCLabelConverter, AttentionLabelConverter
from lib.utils.serialization import load_checkpoint
from lib.evaluation_metrics.metrics import beam_search, get_str_list
from lib.datasets.dataset import Padresize, resizeNormalize
from lib.utils.alphabets import get_alphabets

import socket
import json
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# constants
WINDOW_NAME = "COCO detections"

#功能函数
def clear_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
                
#SRFormer                
def setup_cfg(args):
    # # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg



def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


    
#用于识别的主要函数
def process_image(image_path, model, converter, transform, args):
    try:
        img = Image.open(image_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        torchvision.utils.save_image(img, 'transed_img.jpg')
        
        with torch.no_grad():
            time1 = time.time()
            pred = model.inferrence(img)
            if len(pred) == 2:
                torchvision.utils.save_image(pred[1], 'rectified_img.jpg')
                pred = pred[0]
            
            pred = F.softmax(pred, dim=2)
            score, predicted = pred.max(2)
            pred_score = torch.prod(score, dim=1).cpu().numpy().tolist()
            pred_string = converter.decode(predicted)
            
            time_cost = time.time() - time1
            #print(f'Image: {image_path}')
            #print('Prediction: ', pred_string, 'Prediction Score: ', pred_score, 'Cost time: ', time_cost)
            #print('-' * 50)
            # return {
            #     "image_path": image_path,
            #     "prediction": pred_string,
            #     "score": pred_score[0],  # 假设pred_score是单元素列表
            #     "time_cost": time_cost
            # }   
            return {"prediction": pred_string}
              
            
    except Exception as e:
        print(f'Error processing image {image_path}: {str(e)}')
    


if __name__ == "__main__":
    class Args1:
        config_file = "/root/files/SceneTextDR/configs/SRFormer/Pretrain/R_50_poly.yaml"
        input = ["/root/files/SceneTextDR/received_pic"]  # 注意是列表形式
        output = "/root/files/SceneTextDR/after_detected"
        opts = ["MODEL.WEIGHTS", "/root/files/SceneTextDR/model_weights/ctw1500-srformer-3seg.pth"]
        confidence_threshold = 0.3
        webcam = False
        video_input = None
    args = Args1()
    
    mp.set_start_method('spawn', force=True)  # 在程序开头调用
    originial_picture = '/root/files/SceneTextDR/received_pic'
    cropped_pictures_dir = '/root/files/SceneTextDR/output_cropped'# 裁剪好的照片路径 
    clear_directory(cropped_pictures_dir)
    os.makedirs(cropped_pictures_dir, exist_ok=True)

    #检测部分 得到裁剪结果保存到路径
    # parser = argparse.ArgumentParser(description='SRFormer')
    # parser.add_argument('--config-file', default='/root/DTROcr/SceneTextDR/configs/SRFormer/Pretrain/R_50_poly.yaml', type=str)
    # parser.add_argument('--input', default=originial_picture, type=str)
    # parser.add_argument('--output', default='/root/DTROcr/SceneTextDR/after_detected', type=str)
    # parser.add_argument('--opts', default='/root/DTROcr/SceneTextDR/model_weights/ctw1500-srformer-3seg.pth', type=str)

#python main.py --config-file /root/DTROcr/SceneTextDR/configs/SRFormer/Pretrain/R_50_poly.yaml --input /root/DTROcr/SceneTextDR/received_pic --output /root/DTROcr/SceneTextDR/after_detected --opts MODEL.WEIGHTS /root/DTROcr/SceneTextDR/model_weights/ctw1500-srformer-3seg.pth
    
    mp.set_start_method("spawn", force=True)
    # args = get_parser().parse_args()
    #print(args)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            
            
            polygons = predictions['instances'].polygons.cpu().numpy()

            for i, poly in enumerate(polygons):
                
                # 将坐标点重新组织为(x,y)对
                points = poly.reshape(-1, 2)  # 变为8x2的数组
                #print(points)
                # 找到多边形的最小外接矩形
                rect = cv2.boundingRect(points.astype(np.int32))
    
                # 裁剪矩形区域 (x, y, w, h)
                x, y, w, h = rect
                cropped = img[y:y+h, x:x+w]
    
                # 保存裁剪后的图像
                base_name = os.path.splitext(os.path.basename(path))[0]  # 获取原图文件名（不带扩展名）
                cv2.imwrite(f"/root/files/SceneTextDR/output_cropped/{base_name}_cropped_{i}.jpg", cropped)  # 例如：image1_cropped_0.jpg

            gc.collect()  # 垃圾回收
            torch.cuda.empty_cache()  # 清理 GPU 缓存
            
            # logger.info(
            #     "{}: detected {} instances in {:.2f}s".format(
            #         path, len(predictions["instances"]), time.time() - start_time
            #     )
            # )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    
    #识别算法
    # bash_command = """
    # CUDA_VISIBLE_DEVICES=0,1 python3 /root/files/SceneTextDR/inferrence.py \
    # --image_path /root/files/SceneTextDR/output_cropped/ \
    # --arch ResNet_Scene \
    # --decode_type Attention \
    # --with_lstm \
    # --height 64 \
    # --width 256 \
    # --max_len 50 \
    # --resume /root/files/SceneTextDR/model_path/ASTER/model_best.pth.tar \
    # --alphabets allcases_symbols \
    # --STN_ON \
    # --tps_inputsize 32 64 \
    # --tps_outputsize 32 100 \
    # --tps_margins 0.05 0.05 \
    # --stn_activation none \
    # --num_control_points 20
    # """

    # exit_code = os.system(bash_command)
    # if exit_code != 0:
    #     print("Bash命令执行失败!")
        
    # print("1")

    
    
    class Args2:
        def __init__(self):
            self.image_path = "/root/files/SceneTextDR/output_cropped"
            self.arch = "ResNet_Scene"
            self.decode_type = "Attention"
            self.with_lstm = True
            self.height = 64
            self.width = 256
            self.max_len = 50
            self.resume = "/root/files/SceneTextDR/model_path/ASTER/model_best.pth.tar"
            self.alphabets = "allcases_symbols"
            self.STN_ON = True
            self.tps_inputsize = [32, 64]
            self.tps_outputsize = [32, 100]
            self.tps_margins = [0.05, 0.05]
            self.stn_activation = "none"
            self.num_control_points = 20
            self.punc = False
            self.padresize = False
            self.decoder_sdim = 512  # 需要根据你的模型设置
            self.attDim = 512       # 需要根据你的模型设置
    #args = get_args(sys.argv[1:])
    args = Args2()
    args.alphabets = get_alphabets(args.alphabets)
    if args.punc:
        args.alphabets += " "

    """ Set up model with converter """
    if args.decode_type == 'CTC':
        model = ModelBuilder_CTC(arch=args.arch, rec_num_classes=len(args.alphabets)+1) # +1 for [blank]
        converter = CTCLabelConverter(args.alphabets, args.max_len)
    elif args.decode_type == 'Attention':
        model = ModelBuilder_Att(arch=args.arch,rec_num_classes=len(args.alphabets)+3, #+3 for <EOS>, <PAD>, <UNK>
                sDim=args.decoder_sdim, attDim=args.attDim,max_len_labels=args.max_len,STN_ON=args.STN_ON)
        converter = AttentionLabelConverter(args.alphabets,args.max_len)
    elif args.decode_type == 'DAN': # DAN
        model = ModelBuilder_DAN(arch=args.arch,rec_num_classes=len(args.alphabets)+3, #+3 for <EOS>, <PAD>, <UNK>
                              max_len_labels=args.max_len)
        converter = AttentionLabelConverter(args.alphabets,args.max_len)
    checkpoint = load_checkpoint(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()

    #初始化数组整张图片结果存在数组result
    result = []
    # create transform
    if args.padresize:
        print('using padresize')
        transform = Padresize(args.height, args.width)
    else:
        print('using normal resize')
        transform = resizeNormalize((args.width, args.height))
    
    # check if image_path is a file or directory
    if os.path.isfile(args.image_path):
        # single image processing
        tmp = process_image(args.image_path, model, converter, transform, args)
        
    elif os.path.isdir(args.image_path):
        # batch processing for all images in directory
        print(f'Processing all images in directory: {args.image_path}')
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        
        for filename in sorted(os.listdir(args.image_path)):
            if filename.lower().endswith(supported_formats):
                image_path = os.path.join(args.image_path, filename)
                tmp = process_image(image_path, model, converter, transform, args)
                result.append(tmp)
        result = [item['prediction'][0] for item in result]
        print(result)
        #['Welcomes', 'UTAH', 'You']
    else:
        print(f'Invalid image path: {args.image_path}')
        


