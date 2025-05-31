import java.util.Arrays;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
public class test {
    public static void main(String[] args) throws Exception {
        String language = "zh";

        String filePath = "/root/files/SceneTextDR/language.txt";
 
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            writer.write(language);
        } catch (IOException e) {
            e.printStackTrace();
        }

        String[] pythonArgs = {
            "python3",
            "/root/files/SceneTextDR/main.py",
        };

        System.out.println("执行的命令：" + Arrays.toString(pythonArgs));

        Process proc = Runtime.getRuntime().exec(pythonArgs);
        
        // 创建线程分别读取输出流和错误流（防止阻塞）
        Thread outputThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(proc.getInputStream()))) {
                String line;
                System.out.println("=== Python 输出 ===");
                while ((line = reader.readLine()) != null) {
                    System.out.println(line);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        
        Thread errorThread = new Thread(() -> {
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(proc.getErrorStream()))) {
                String line;
                System.err.println("=== Python 错误 ===");
                while ((line = reader.readLine()) != null) {
                    System.err.println(line);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        
        outputThread.start();
        errorThread.start();
        
        // 等待进程结束
        int exitCode = proc.waitFor();
        outputThread.join();
        errorThread.join();
        
        System.out.println("Python 进程退出码：" + exitCode);
    }
}