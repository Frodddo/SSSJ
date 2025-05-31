import java.net.ServerSocket;
import java.net.Socket;
import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.util.concurrent.*;

//服务器端

public class Server {
    private static final String DELIMITER = "SECOND_FILE_STARTS_HERE";
    public static void main(String[] args) throws Exception {
        //1.准备连接
        ServerSocket server = new ServerSocket(8892);
        System.out.println("等待连接");
        int count = 0;
        while(true){
            Socket socket = server.accept();
            System.out.println("第" + ++count + "个客户端"+socket.getInetAddress().getHostAddress()+"连接成功！！");
            ClientHandlerThread ct = new ClientHandlerThread(socket);
            ct.start();
            if(ClientHandlerThread.status == 1){
                server.close();
            }
        }
    }

    static class ClientHandlerThread extends Thread{
        static private int status = 0;
        private Socket socket;
        private String ip;

        public ClientHandlerThread(Socket socket){
            super();
            this.socket = socket;
            this.ip = socket.getInetAddress().getHostAddress();
        }

        private static void sendFile(String filename, OutputStream out) throws IOException {
        Path filePath = Paths.get(filename);
        if (Files.exists(filePath)) {
            byte[] fileContent = Files.readAllBytes(filePath);
            out.write(fileContent);
            out.flush();
            System.out.println("已发送文件: " + filename);
        } else {
            System.out.println("文件不存在: " + filename);
        }
    }

        @Override
        public void run() {
            try {
                String filePath1 = "/root/files/SceneTextDR/translated/translated.txt"; // 替换为你的文件路径
                String filePath2 = "/root/files/SceneTextDR/translated/deepseek.txt"; // 替换为你的文件路径
                String filePath3 = "/root/files/rec_results/rec_results.txt"; // 替换为你的文件路径


                // 创建 File 对象
                File f1 = new File(filePath1);
                File f2 = new File(filePath2);
                File f3 = new File(filePath3);
                if (f1.exists()){
                    f1.delete();
                }
                if (f2.exists()){
                    f2.delete();
                }
                if (f3.exists()){
                    f3.delete();
                }

                //获取socket流输入内容（照片与翻译语言）
                InputStream is = socket.getInputStream();
                DataInputStream dis = new DataInputStream(is);

                String str = dis.readUTF();
                String filePath = "/root/files/SceneTextDR/language.txt";
 
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
                    writer.write(str);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                System.out.println("接收到的信息: " + str);
                if(str == "bye"){
                    status = 1;

                }

                FileOutputStream fos = new FileOutputStream("/root/files/SceneTextDR/received_pic/received.jpg");
                int len = 0;
                byte[] bytes = new byte[1024];
                while((len = is.read(bytes)) != -1){
                    fos.write(bytes,0,len);
                }
                System.out.println("图片接受成功！");


                //调用python文件
                //python3 main.py --config-file /root/files/SceneTextDR/configs/SRFormer/Pretrain/R_50_poly.yaml --input /root/files/SceneTextDR/received_pic --output /root/files/SceneTextDR/after_detected --opts MODEL.WEIGHTS /root/files/SceneTextDR/model_weights/ctw1500-srformer-3seg.pth
                String[] args1 = new String[] { "python3", "/root/files/SceneTextDR/main.py" };
                //String[] args1 = new String[] { "python3", "/home/ubuntu/finaltorchafter/javatest.py"};//第二个为python脚本所在位置，后面的为所传参数（得是字符串类型）
                System.out.println("输入的语句为：" + args1.toString());
                Process proc = Runtime.getRuntime().exec(args1);// 执行py文件

                BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream(),"gb2312"));//解决中文乱码，参数可传中文
                String line = null;
                while ((line = in.readLine()) != null) {
                    System.out.println(line);
                }
                in.close();
                proc.waitFor();
                socket.shutdownInput();

                String filePath4 = "/root/files/SceneTextDR/translated/translated.txt"; // 替换为你的文件路径
                String filePath5 = "/root/files/rec_results/rec_results.txt"; // 替换为你的文件路径
                String filePath6 = "/root/files/SceneTextDR/translated/deepseek.txt"; // 替换为你的文件路径

                File f4 = new File(filePath4);
                File f5 = new File(filePath5);
                File f6 = new File(filePath6);

                String filePath_ = null;
                if (f4.exists()){
                    filePath_ = filePath4;
                }
                else if (f5.exists()){
                    filePath_ = filePath5;
                }

            
                OutputStream out = socket.getOutputStream();
                PrintWriter writer = new PrintWriter(out, true);
                //立即发送第一个文件 运行到此处python已运行完毕
                sendFile(filePath_, out);
                writer.println(DELIMITER);
                sendFile(filePath6, out);




                
                System.out.println("结果返回成功！");
                socket.shutdownOutput();

/*                String output = new String(cbuf,0,len2);//char
                System.out.print(output);*/


            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            } finally {
                if(socket != null){
                    try {
                        socket.close();
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                }

            }


        }
    }

}


