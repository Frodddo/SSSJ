import java.io.*;

public class Main {
    public static void main(String[] args) {
        try {
            // 启动 Python 脚本
            Process process = Runtime.getRuntime().exec("python example.py");

            // 等待“开始写入”标志文件消失
            while (new File("flag.lock").exists()) {
                Thread.sleep(100); // 等待文件被删除
            }

            // 读取主标志文件
            BufferedReader reader = new BufferedReader(new FileReader("flag.txt"));
            String flag = reader.readLine();
            reader.close();

            // 处理结果
            if ("PROCESS_SUCCESS".equals(flag)) {
                System.out.println("Python process succeeded!");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}