import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import numpy as np
import deepl 

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
from openai import OpenAI
import subprocess


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
            print(f'Image: {image_path}')
            print('Prediction: ', pred_string, 'Prediction Score: ', pred_score, 'Cost time: ', time_cost)
            print('-' * 50)
            # return {
            #     "image_path": image_path,
            #     "prediction": pred_string,
            #     "score": pred_score[0],  # 假设pred_score是单元素列表
            #     "time_cost": time_cost
            # }   
            return {"prediction": pred_string}
              
            
    except Exception as e:
        print(f'Error processing image {image_path}: {str(e)}')
    

#检测算法
def detection():
    class Args1:
        #config_file = "/root/files/SceneTextDR/configs/SRFormer/Pretrain/R_50_poly.yaml"
        config_file = "/root/files/SceneTextDR/configs/SRFormer/Pretrain_ArT/R_50_poly.yaml"
        input = ["/root/files/SceneTextDR/received_pic"]  # 注意是列表形式
        output = "/root/files/SceneTextDR/after_detected"
        #opts = ["MODEL.WEIGHTS", "/root/files/SceneTextDR/model_weights/ctw1500-srformer-3seg.pth"]
        opts = ["MODEL.WEIGHTS", "/root/files/SceneTextDR/model_weights/art-srformer-1seg.pth"]
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
            print("-------------------")
            print(polygons)
            print("-------------------")
            for i, poly in enumerate(polygons):
                
                # 将坐标点重新组织为(x,y)对
                points = poly.reshape(-1, 2)  # 变为8x2的数组
                print(points)
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
            
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

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
    return polygons
#识别算法
def recognition_english():
    #参数初始化    
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
            #self.resume = "/root/files/SceneTextDR/model_weights/art-srformer-1seg.pth"
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
        return result
        #['Welcomes', 'UTAH', 'You']
    else:
        print(f'Invalid image path: {args.image_path}')


def recognition_chinese():
    cmd = [
    "python3",
    "/root/files/OpenOCR/tools/infer_rec.py",
    "--c", "/root/files/OpenOCR/configs/rec/svtrv2/repsvtr_ch.yml",  # 参数和值分开
    "--o", "Global.infer_img=/root/files/SceneTextDR/output_cropped/"
]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout) 
    
    file_path = "/root/files/rec_results/rec_results.txt"
    text_list = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")  # 按制表符分割
            if len(parts) >= 2:  # 确保至少有2列
                text_list.append(parts[1])  # 提取识别文本

    # 打印结果
    print(text_list)
    return(text_list)
    
    
def translation(result,dlanguage):
    auth_key = "7aa91bae-3951-4e8b-a396-6c9da0b3c842:fx"  # Replace with your key
    translator = deepl.Translator(auth_key)
    
    # 如果result是列表，先转换为字符串（用空格连接）
    if isinstance(result, list):
        text_to_translate = ' '.join(result)
    else:
        text_to_translate = result
    # Read and quote text from results file
    translated_text = translator.translate_text(text_to_translate, target_lang=dlanguage)
    # 文件路径
    file_path = '/root/files/SceneTextDR/translated/translated.txt'
 
    # 调用函数将翻译后的文本写入文件
    write_translated_text_to_file(translated_text.text, file_path) 
    print(translated_text.text)
    return translated_text

def translate_word_by_word(result, dlanguage):
    auth_key = "7aa91bae-3951-4e8b-a396-6c9da0b3c842:fx"  # Replace with your key
    translator = deepl.Translator(auth_key)
    
    # 如果result是列表，先转换为字符串（用空格连接），然后拆分为单词列表
    if isinstance(result, list):
        words = ' '.join(result).split()
    else:
        words = result.split()
    
    translated_words = []
    for word in words:
        # 对每个单词进行翻译
        try:
            translated_word = translator.translate_text(word, target_lang=dlanguage).text
            translated_words.append(translated_word)
        except Exception as e:
            print(f"Error translating word '{word}': {e}")
            translated_words.append(word)  # 如果翻译失败，保留原单词
    
    # 将翻译后的单词重新组合成字符串
    # translated_text = ' '.join(translated_words)
    print(translated_words)

 
    # 文件路径
    file_path = '/root/files/SceneTextDR/translated/translated.txt/root/files/SceneTextDR/translated/translated.txt'
 
    # 调用函数将翻译后的文本写入文件
    write_translated_text_to_file(translated_text, file_path) 
    return translated_text
    
def write_translated_text_to_file(translated_text, file_path):
    try:
        # 检查文件是否存在，如果存在则删除
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Existing file {file_path} has been deleted.")
        
        # 写入新的翻译文本
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(translated_text)
        print(f"Translated text successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")
 
     
def llm(polygons,result,language):
    # 初始化OpenAI客户端
    client = OpenAI(
        # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
        api_key = "sk-c62c80ccd9d94857b36e4f3f25d49a9d",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )   
    reasoning_content = ""  # 定义完整思考过程
    answer_content = ""     # 定义完整回复
    is_answering = False   # 判断是否结束思考过程并开始回复

    # 创建聊天完成请求
    completion = client.chat.completions.create(
        model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称
        messages=[
            {"role": "user", "content": f"现在我已经在照片中识别好了文字 需要你结合我坐标框和坐标框内的文字来进行理解 你有坐标信息了 根据坐标信息得到单词之间的相对位置 从而模拟人眼阅读的顺序(从左到右从上到下) 优先把左上方的单词放在句首 在此基础上组合 帮我理解并输出正常语序 组成词组或者句子 每个框上下各有四个点的坐标为{polygons} 每个框对应的单词为{result} 仔细思考给我最可能的组合方式 并翻译成{language}语言 只需要告诉我最有可能的组合和翻译之后的结果 中间过程不要 只输出两个结果不要多余的信息 两个之间空一行"}
        ],
        stream=True,
        # 解除以下注释会在最后一个chunk返回Token使用量
        # stream_options={
        #     "include_usage": True
        # }
    )

    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

    for chunk in completion:
        # 如果chunk.choices为空，则打印usage
        if not chunk.choices:
            print("\nUsage:")
            print(chunk.usage)
        else:
            delta = chunk.choices[0].delta
            # 打印思考过程
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                print(delta.reasoning_content, end='', flush=True)
                reasoning_content += delta.reasoning_content
            else:
                # 开始回复
                if delta.content != "" and is_answering == False:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                    is_answering = True
                # 打印回复过程
                print(delta.content, end='', flush=True)
                answer_content += delta.content
    print("最后完整思考过程"+"\n")
    print(reasoning_content)
    print("最后完整回复"+"\n")
    print(answer_content)
    # 文件路径
    file_path = '/root/files/SceneTextDR/translated/deepseek.txt'
 
    # 调用函数将翻译后的文本写入文件
    write_translated_text_to_file(answer_content, file_path) 


if __name__ == "__main__":
    #初始化两个files
    with open("flag1.lock", "w") as f:
        f.write("WRITING")
    with open("flag2.lock", "w") as f:
        f.write("WRITING")
    
    file_path = "/root/files/SceneTextDR/language.txt"
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            strr = file.read()
    except Exception as e:
        print(f"读取文件时出错: {e}")
    parts = strr.split(',')

    print(parts)  
    source = parts[0]
    language = parts[1]
    print("source_language:", source)  
    print("dest_language:", language)
    
    #检测
    polygons = detection()
    result = []
    #识别中文
    if source == 'zh':
        #识别获得文字结果
        result = recognition_chinese()
    if source == 'en':
        result = recognition_english()
    print("-----------------")
    print(result)
    #翻译
    translation(result,language)
    # print("finished")
    # print(polygons)
    # print(result)
    
    #此时已经完成了files1的写操作
    os.remove("flag1.lock")
    
    llm(polygons,result,language)
    #此时已经完成了files2的写操作
    os.remove("flag2.lock")

                
    
    
