import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import org.apache.hadoop.util.GenericOptionsParser;

public class KNN_MapReduce {
	/*KNN mapreduce实现*/

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (otherArgs.length != 3) {
            System.err.println("Usage: KNN_MapReduce <trainSet_path> <testSet_path> <output_path>");
            System.exit(2);
        }
		
		//若存在output文件夹，则先删除output文件夹
		FileSystem fileSystem = FileSystem.get(conf);
        if (fileSystem.exists(new Path(otherArgs[2])))
        {
            fileSystem.delete(new Path(otherArgs[2]), true);
        }
		
		//设置基本信息
        Job job = new Job(conf, "KNN");
        job.setJarByClass(KNN_MapReduce.class);
        job.setInputFormatClass(TextInputFormat.class);

        //设置Mapper
        job.setMapperClass(KNN_Mapper.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setNumReduceTasks(1);
        job.setPartitionerClass(HashPartitioner.class);

        //设置Reducer
        job.setReducerClass(KNN_Reducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);

        //设置训练数据的输入路径
        FileInputFormat.addInputPath(job, new Path(otherArgs[1]));
        //设置任预测结果的输出路径
        FileOutputFormat.setOutputPath(job, new Path(otherArgs[2]));
		//等待任务完成后退出程序
        System.exit(job.waitForCompletion(true) ? 0 : 1);

    }

    public static class KNN_Mapper extends Mapper<LongWritable, Text, Text, Text> {
        public ArrayList<Instance> trainSet = new ArrayList<Instance>();
		/*******在这里修改 K 值*******/
		public int K = 1;
		/*******在这里修改 K 值*******/
        protected void setup(Context context) throws IOException, InterruptedException {
			//读取训练集
            FileSystem fileSystem = null;
            try {
                fileSystem = FileSystem.get(new URI("hdfs://192.168.142.128:9000/"), new Configuration());
            } catch (Exception e) {
            }
            FSDataInputStream trainSet_input = fileSystem.open(new Path("hdfs://192.168.142.128:9000/knn_train/train.txt"));
            BufferedReader trainSet_data = new BufferedReader(new InputStreamReader(trainSet_input)); 

			//逐行划分训练集中特征以及标签
            String str = trainSet_data.readLine();
            while (str != null) {
                trainSet.add(new Instance(str));
                str = trainSet_data.readLine();
            }
        }

        protected void map(LongWritable k1, Text v1, Context context) throws IOException, InterruptedException {
			
            ArrayList<Double> distance = new ArrayList<Double>(K);
            ArrayList<String> trainlabel = new ArrayList<String>(K);
			
			//初始化前 K 小距离和标签值
            for (int i = 0; i < K; i++)
            {
                distance.add(Double.MAX_VALUE);
                trainlabel.add("NAN");
            }
			
			//读取一个test样本
            Instance testInstance = new Instance(v1.toString());
			//计算test样本和各个train样本距离
            for (int i = 0; i < trainSet.size(); i++) {
                double dis = Distance.calcEuclideanDistance(trainSet.get(i).getAttributeSet(), testInstance.getAttributeSet());

                for (int j = 0; j < K; j++)//若距离比元素值小，则覆盖
                {
                    if (dis < (Double) distance.get(j)) {
                        distance.set(j, dis);
                        trainlabel.set(j, trainSet.get(i).getlabel() + "");
                        break;
                    }
                }
            }
			
			//mapper过程输出: 以测试集特征为 key 值，以 K 个近邻的标签值列表为 value 值
            for (int i = 0; i < K; i++)
            {
                context.write(new Text(v1.toString()), new Text(trainlabel.get(i) + ""));
            }
        }
    }

    public static class KNN_Reducer extends Reducer<Text, Text, Text, NullWritable> {
		 /*多数投票函数实现*/
        protected void reduce(Text k2, Iterable<Text> v2s, Context context) throws IOException, InterruptedException {
			
			//提取输入的标签值
            ArrayList<String> KNeighborsLabel = new ArrayList<String>();
            for (Text v2 : v2s)
            {
                KNeighborsLabel.add(v2.toString());
            }
			
			//统计 K 近邻的标签
            String predictlabel = MajorityVoting(KNeighborsLabel);
			
			//reducer过程输出： 以测试集特征以及预测标为 key 值，以 空值 为 value 值
            String preresult = k2.toString() + "," + predictlabel;
            context.write(new Text(preresult), NullWritable.get());
        }

        public String MajorityVoting(ArrayList KNeighbors) {

            HashMap<String, Double> freqCounter = new HashMap<String, Double>();
			
			//遍历所有输入标签，统计出现次数
            for (int i = 0; i < KNeighbors.size(); i++)
            {
                if (freqCounter.containsKey(KNeighbors.get(i))) {
                    double frequence = freqCounter.get(KNeighbors.get(i)) + 1;
                    freqCounter.remove(KNeighbors.get(i));
                    freqCounter.put((String) KNeighbors.get(i), frequence);
                } else {
					freqCounter.put((String) KNeighbors.get(i), new Double(1));
				}
            }
			
            //多数投票得到最终预测标签
            Iterator it = freqCounter.keySet().iterator();
            double maxi = Double.MIN_VALUE;
            String final_predict = null;
            while (it.hasNext())//取出现最多的标签
            {
                String key = (String) it.next();
                Double labelnum = freqCounter.get(key);
                if (labelnum > maxi) {
                    maxi = labelnum;
                    final_predict = key;
                }
            }
            return final_predict;
        }
    }
}

class Distance {
	/*计算样本a与样本b之间的欧式距离*/
    public static double calcEuclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }
}

class Instance {
	/*根据每一行数据划分数据的特征、标签*/
    private double[] attributeSet;//样例属性
    private double label;         //样例标签

    public Instance(String data_line) {
		//用逗号分隔数据
        String[] data_input = data_line.split(",");		 
		//前length-1项为属性
        attributeSet = new double[data_input.length - 1];
        for (int i = 0; i < attributeSet.length; i++) {
            attributeSet[i] = Double.parseDouble(data_input[i]);
        }
		//第length项为标签
        label = Double.parseDouble(data_input[data_input.length - 1]);
    }

    public double[] getAttributeSet() {
        return attributeSet;
    }

    public double getlabel() {
        return label;
    }
}