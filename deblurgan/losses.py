import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np

# Note the image_shape must be multiple of patch_shape
image_shape = (256, 256, 3)


def l1_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def perceptual_loss_100(y_true, y_pred):
    return 100 * perceptual_loss(y_true, y_pred)


# 感知损失  -> 用于生成器
# 使用VGG16分别提取生成图片和真实图片的特征，比较的block3_conv3层的输出
# 最后返回的损失是生成图片和真实图片之间特征差的平方再取均值
def perceptual_loss(y_true, y_pred):
    '''
    参数：
    include_top:是否保留顶层的3个全连接网络
    weights：None代表随机初始化，即不加载预训练权重。'imagenet'代表加载预训练权重
    input_tensor：可填入Keras tensor作为模型的图像输出tensor
    input_shape：可选，仅当include_top=False有效，应为长为3的tuple，指明输入图片的shape，图片的宽高必须大于
                 48，如(200,200,3)

    返回值：
    pooling：当include_top=False时，该参数指定了池化方式。None代表不池化，最后一个卷积层的输出为4D张量。
            ‘avg’代表全局平均池化，‘max’代表全局最大值池化。
    classes：可选，图片分类的类别数，仅当include_top=True并且不加载预训练权重时可用
    '''
    vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

'''
传统Loss造成GAN训练困难的原因：
因为真实样本的概率分布与生成器生成的样本概率分布的支撑集不同，又由于两者的流型
的维度皆小于样本空间的维度，因而两者的流型基本上是不可能完全对齐的，因而即便有少量相交
的点，它们在两个概率流型上的测度为0，可忽略，因而可以将两个概率的流型看成是可分离的，
因而若是一个最优的判别器去判断则一定可以百分百将这两个流型分开，即无论我们的生成器如何
努力皆获得不了分类误差的信息
'''
# 对整个模型(生成器+判别器)的输出执行的是wasserstein损失，它取的是生成图片和真实图片之间
# 差异的均值，该损失可以改善生成对抗网络的收敛性
# Wasserstein 距离，推土机距离，用来表示两个分布的相似程度。衡量了把数据从分布“移动成”分布时
#                   所需要移动的平均距离的最小值
# 该距离的优点是：
# 1、若两个概率分布完全重合时，W_Distance=0
# 2、是对称的
# 3、即使两个分布的支撑集没有重叠或者重叠非常少，亦可以衡量两个分布的远近，并在满足一定
#    条件下可微，具备了后向传输的能力。


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true*y_pred)


def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))

    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)

    return K.mean(gradient_penalty)
