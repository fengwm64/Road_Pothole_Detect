# Road Pothole Detect 坑洼道路检测

该仓库包含了我参加MatherCup_BigData_2023竞赛A题的所有代码和最终报告，专注于坑洼道路的分类问题。基于深度学习理论，我调整了预训练的VGG16和Inception_v3模型的卷积神经网络结构，建立了一个在自动检测道路图像中是否存在坑洼现象时能够实现高准确率、快速处理速度和精确分类的模型。

![](https://cdn.jsdelivr.net/gh/oixel64/imgs/imgs/202406231803929.jpg)

## 工作内容

### 数据预处理：

清洗数据集以删除特征不清晰的图片。由于样本数量有限，采用了图像随机缩放、剪切、旋转、垂直和水平翻转、亮度调整以及加入椒盐噪声等数据增广方法，以丰富数据集的多样性。

### 模型架构：

- **Advanced-VGG16**：在VGG16基础上改进了全连接层。
- **Advanced-Inception_v3**：将激活函数改为sigmoid，形成改进后的Inception_v3模型，以更好地捕捉图像中的坑洼特征，提升道路坑洼的识别性能。

初步训练表明，Advanced-Inception_v3模型在更少的迭代次数下提升了准确率，相同迭代次数下准确率更高，训练开销更小。因此选择该模型作为坑洼道路的分类器。

### 模型优化：

对选定的Advanced-Inception_v3分类模型进行多次训练，优化损失函数选择、优化器选择、是否添加全局平均池化层、是否添加正则化、是否添加早停回调函数等五个维度，最终分类模型的分类准确率达到了93%。

### 评估：

通过分类准确率、召回率、查准率、F1分数和Kappa系数等评估指标表明，该模型具有较强的性能。特别地，坑洼图像的查准率达到96%，高于正常图像的91%，进一步验证了该分类模型在道路坑洼检测中的有效性和鲁棒性。

### 测试：

使用训练过程中获得的优秀模型参数对提供的测试集进行预测。在Colab平台上加载模型参数，并进行一系列预测，结果保存在名为`test_result.csv`的文件中。

## 关键词：
深度学习、VGG16模型、Inception_v3模型、卷积神经网络、坑洼检测

---

This repository contains all the code and the final report submitted for Question A of MatherCup_BigData_2023 competition, focusing on road pothole classification. Based on deep learning theory, I adjusted the convolutional neural network structures of pretrained VGG16 and Inception_v3 models to build a model that achieves high accuracy, fast speed, and precise classification for automatically detecting potholes in road images.

## My Work

### Data Preprocessing:

The dataset was cleaned to remove images with unclear features. Due to the limited number of samples, data augmentation techniques including random scaling, cropping, rotation, vertical and horizontal flipping, brightness adjustment, and adding salt-and-pepper noise were applied to enrich dataset diversity.

### Model Architecture:

- **Advanced-VGG16**: Enhanced the fully connected layers on the base of VGG16.
- **Advanced-Inception_v3**: Modified the activation function to sigmoid, forming an improved Inception_v3 model to better capture pothole features in images and enhance road pothole detection performance.

Initial training showed that the Advanced-Inception_v3 model achieved faster accuracy improvement with fewer iterations, higher accuracy with the same number of iterations, and reduced training overhead. Therefore, this model was selected as the pothole road classifier.

### Model Optimization:

The selected Advanced-Inception_v3 classification model underwent multiple training sessions, optimizing parameters across dimensions such as loss function selection, optimizer choice, inclusion of global average pooling layer, regularization, and early stopping callback. The final model achieved a classification accuracy of 93%.

### Evaluation:

Evaluation metrics including accuracy, recall, precision, F1 score, and Kappa coefficient demonstrated strong performance. Notably, the precision for pothole images reached 96%, higher than the 91% precision for normal images, further confirming the effectiveness and robustness of the proposed classification model in road pothole detection.

### Testing:

Using the well-tuned model parameters obtained during training, predictions were made on the provided test set. The model parameters were loaded on the Colab platform for prediction, and results were saved in a file named `test_result.csv`.

## Keywords:
Deep Learning, VGG16 Model, Inception_v3 Model, Convolutional Neural Network, Pothole Detection




