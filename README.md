# peron_attribute
training person attribute with mxnet
## 算法流程
1. 利用 market-1501 和 duke 两个行人reid 数据集中的属性标注，训练人体属性模型
2. 训练框架使用 MXNET
3. 训练的基本流程是：读取图片生成recordio -> 训练 -> 模型测试
4. 现在用的基本模型是 res-18 imagenet 预训练模型

## 基本类别

|大类|类别|类别数|详细类别|
|---|---|---|---|
|性别|gender|2|0 male<br> 1 female|
|附属物品|hat|2|0 no_hat<br>1 hat|
|附属物品|bag|2|0 no_bag<br>1 bag|
|附属物品|handbag|2|0 no_handbag<br>1 handbag|
|附属物品|backpack|2|0 no_backpack<br>1 backpack|
|衣着颜色|updress|8|0 upunknown <br>1 upblack<br>2 upblue <br>3 upgreen<br>4 upgray<br>5 uppurple<br>6 upred<br>7 upyellow'|
|衣着颜色|downdress|11|0 downunknown<br>1 downblack<br>2 downblue<br>3 downbrown<br>4 downgray<br>5 downgreen<br>6 downpink<br> 7 downpurple<br>8 downwhite<br>9 downred<br>10 downyellow|

## P100K的推理结果

http://otr41gcz3.bkt.clouddn.com/P100K_save.json


## TODOliST
- [x] 调通基本训练框架，使用多分支fc得到多个属性结果
- [x] 加入一些训练技巧，比如dropout，调整学习率等
- [ ] 将基本模型作用到P100K 数据集，做二次打标过滤，扩充数据集
- [ ] 利用扩充的数据集迭代训练
- [ ] 结合基本人体检测模型，实际测试效果
