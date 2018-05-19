# cnn_mnist
使用简单的卷积神经网络实现手写数字识别。
simple apply by training mnist to recognize handwritten digits, based on cnn.

只上传了python程序。
only python files are uploaded.

实验环境：GTX 950M + 2G显存
        anaconda3-4.2.0(pyhton3.5) + cuda + cudnn + tensorflow-gpu 1.3.0

原本在程序中设置了显存按需增长模式，但我可怜的机器依旧无法同时进行训练和测试，因此删去了设置部分。

训练集和测试集为mnist数据集，对自己手写的数字进行识别。

模块化：

forward.py实现前向传播过程，网络结构为卷积→激活→池化→卷积→激活→池化→全连接层。

卷积层1的卷积核大小为5×5，共32个卷积核；
卷积层2的卷积核大小为5×5，共64个卷积核；
两池化层卷积核均为2×2，最大值池化；
卷积层和池化层均进行了全零填充；
两激活层均采用线性激活，即使用relu函数作为激活函数。

fc层的隐藏层包含1个隐藏层，该隐藏层包含512个节点。


backward.py实现反向传播过程，使用方法如下：

learning rate的衰减方式采取指数衰减
loss function采用交叉熵结合softmax
optimizer为梯度下降方法


test.py实现使用mnist自带的测试集对模型进行评价。

由于我的机器的内存太小，无法一次读入测试集所有数据，测试集也分批喂入数据进行测试。


apply.py实现对自己手写的数字进行预处理后送入训练好的模型进行识别。
