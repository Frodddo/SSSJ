import os
import time
 
def process_data():
    # 创建“开始写入”标志文件
    with open("flag.lock", "w") as f:
        f.write("WRITING")
 
    # 模拟耗时操作
    time.sleep(5)
 
    # 写入主标志文件
    with open("flag.txt", "w") as f:
        f.write("PROCESS_SUCCESS")
 
    # 删除“开始写入”标志文件
    os.remove("flag.lock")
 
if __name__ == "__main__":
    process_data()