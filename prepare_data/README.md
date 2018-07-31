## 样本准备步骤
1. 下载market 和 duke 的属性标注文件   
  训练：http://otr41gcz3.bkt.clouddn.com/attr_label.json 测试： http://otr41gcz3.bkt.clouddn.com/attr_label_test.json
2. 运行 creatt_list.upynb 生成对应的lst 文件
3. 执行 python im2rec.py 生成对应的recordio 文件 注意需要安装mxnet和 制定 --pack-label 参数和 resize 参数 resize大小的参数目前在代码里面写死了，需要修改
