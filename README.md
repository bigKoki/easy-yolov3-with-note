# easy-yolov3-with-note
coding by huangxinyu

基本的yolov3代码，带有我编码时写的详细注释，适合初学者学习。

另外在我的博客有对yolov3的解读：http://dayefuzi.cn/post/68

训练：

1.将格式化的数据集放在data文件夹下，images里放图像，annotations/annotations.txt存放图像名和box的坐标和类别
 
  如：4.jpg 177,18,229,80,0 266,26,313,78,0

2.在config/yolov3.py里修改自己想要的训练参数

3.开始训练
