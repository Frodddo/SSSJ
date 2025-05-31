

import subprocess
if __name__ == "__main__":
    cmd = [
    "python3",
    "/root/files/OpenOCR/tools/infer_rec.py",
    "--c", "/root/files/OpenOCR/configs/rec/svtrv2/repsvtr_ch.yml",  # 参数和值分开
    "--o", "Global.infer_img=/root/files/SceneTextDR/output_cropped/"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout) 