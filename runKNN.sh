#!/bin/bash
#编译java类
javac KNN_MapReduce.java
ls
#打包jar包
jar -cvf KNN_MapReduce.jar *.class
rm *.class
rm *.java
#运行 mapreduce
/usr/local/hadoop/bin/hadoop jar KNN_MapReduce.jar KNN_MapReduce /knn_train /knn_test /knn_output
rm *.jar
#导出运行结果文件
/usr/local/hadoop/bin/hdfs dfs -get /knn_output/*
sz part-r-00000
rm part-r-00000
rm _SUCCESS
