# pytorch-feed-forward-network-test
个简单的前馈神经网络(feed-forward network）。它接受一个输入，然后将它送入下一层，一层接一层的传递，最后给出输出。  一个神经网络的典型训练过程如下：  定义包含一些可学习参数(或者叫权重）的神经网络 在输入数据集上迭代 通过网络处理输入 计算loss(输出和正确答案的距离） 将梯度反向传播给网络的参数 更新网络的权重，一般使用一个简单的规则：weight = weight - learning_rate * gradient

参考[神经网络](https://pytorch.apachecn.org/docs/1.4/blitz/neural_networks_tutorial.html)    

可以使用`torch.nn`包来构建神经网络.
我们已经介绍了`autograd`包，`nn`包则依赖于`autograd`包来定义模型并对它们求导。一个`nn.Module`包含各个层和一个`forward(input)`方法，该方法返回`output`。
例如，下面这个神经网络可以对数字进行分类：  
![LeNet-feed-forward-network](https://pytorch.org/tutorials/_images/mnist.png)

一个神经网络的典型训练过程如下：

  ·定义包含一些可学习参数(或者叫权重）的神经网络
  ·在输入数据集上迭代
  ·通过网络处理输入
  ·计算loss(输出和正确答案的距离）
  ·将梯度反向传播给网络的参数
  ·更新网络的权重，一般使用一个简单的规则：`weight = weight - learning_rate * gradient`

##定义网络
让我们定义这样一个网络：
    
    '''python  
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
    
        def __init__(self):
            super(Net, self).__init__()
            # 输入图像channel：1；输出channel：6；5x5卷积核
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
    
        def forward(self, x):
            # 2x2 Max pooling
            x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
            # 如果是方阵,则可以只使用一个数字进行定义
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
        def num_flat_features(self, x):
            size = x.size()[1:]  # 除去批处理维度的其他所有维度
            num_features = 1
            for s in size:
                num_features *= s
            return num_features
    
    
    net = Net()
    print(net)
    '''
