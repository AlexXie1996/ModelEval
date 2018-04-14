# ModelEval
评估模型的一些常用指标

## 环境

  - python3.6 + numpy + pandas
  
## 实现

  - 对一些评估模型（变量或者预测）的指标的一些实现，定义在udumbara.py中，包括
  
    + WOE
    + IV
    + PSI
    + KS
    + ROC
    + AUC
    
  - main.py提供一份测试代码，数据在test.csv中
  
## 使用

  - 可以但不限于使用read_csv() 将数据读入并分出x和y
  - 可以但不限于使用discrete.py的Discreter将x变量分箱
  - 指标的调用接口有两种，例如IV() 和 IV_from_suit()，接受的参数有所不同：
  
    - packet：5D-array-like，[[suit0], [suit1], ..]
    - suit: 4D-array-like,[[box0],[box1], ..]
    - box: 3D-array-like, [[sample0], [sample1], ..]
    - sample: 2D-array-like, [[x0, y0], [x1, y1], ..]
    
  - ks值有verbose选项，分别详细输出ks值与max(ks)
 
 ## 吐槽
 
  - 好吧我承认我没有写注释，接口又难看
