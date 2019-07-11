from keras.layers import Input, Activation, Add, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from .layer_utils import ReflectionPadding2D, res_block

# the paper defined hyper-parameter:chr
channel_rate = 64

# image_shape必须是patch_shape的倍数
image_shape = (256, 256, 3)
patch_shape = (channel_rate, channel_rate, 3)

ngf = 64
ndf = 64
input_nc = 3
output_nc = 3

# 输入到生成器中的图像shape
input_shape_generator = (256, 256, input_nc)
# 输入到判别器中的图像shape
input_shape_discriminator = (256, 256, output_nc)
# 残差块的个数
n_blocks_gen = 9

# 生成器会用模糊图像生成清晰图像，然后生成器生成的图像会输入判别器中，判别器会将图像和
# 训练集中的真实图像进行对比。判别器返回一个介于0(假图像)和1(真图像)之间的数字。


# 生成器将模糊图像作为输入，输出一个假的去模糊图像，它内置有反卷积层。
def generator_model():
    """Build generator architecture."""

    # 创建一个keras tensor， shape为: ? x 256 x 256 x 3
    inputs = Input(shape=image_shape)
    # 对输入图像做边界扩展，宽高各扩展6个像素，当前shape为: ? x 262 x 262 x 3
    x = ReflectionPadding2D((3, 3))(inputs)
    # 卷积操作，卷积核大小7x7，卷积方式valid，卷积核个数64，当前shape为: ? x 256 x 256 x 64
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
    # 对卷积层输出的特征图进行均值和方差的规范化，减少梯度消失、梯度爆炸
    x = BatchNormalization()(x)
    # 使用relu函数进行一次激活操作
    x = Activation('relu')(x)

    # 设置2次下采样
    n_downsampling = 2
    # 进行2次same方式的卷积操作，每次卷积特征图大小减半，并对特征图进行归一化和激活函数处理，
    # 两次卷积过程中，shape变化：?*256*256*64 -> ?*128*128*128 -> ?*64*64*256
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    # mult为4，残差块中卷积操作中卷积核个数为64*4=256
    mult = 2**n_downsampling
    # 定义9层残差块，最后输出的是特征图大小为 ?*64*64*256
    for i in range(n_blocks_gen):
        x = res_block(x, ngf*mult, use_dropout=True)

    # 两次上采样，特征图shape变化如下：?*64*64*256 -> ?*128*128*128 -> ?*256*256*64
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        # x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3, 3), strides=2, padding='same')(x)
        # 进行上采样，默认宽高都放大2倍
        x = UpSampling2D()(x)
        x = Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)    # 使用relu激活函数进行处理

    # 宽高各扩展6，当前shape为 ?*262*262*64
    x = ReflectionPadding2D((3, 3))(x)
    # 进行一次卷积操作，卷积核个数为3，shape变化：?*262*262*64 -> ?*256*256*3
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='valid')(x)
    # 使用Tanh函数进行激活
    x = Activation('tanh')(x)

    # 将输入和输出连接起来，输入inputs的shape为：?*256*256*3，输出x的shape为：?*256*256*3
    outputs = Add()([x, inputs])
    # 将输入和输出的连接除以2，来保持输出归一化
    outputs = Lambda(lambda z: z/2)(outputs)

    # 利用输入和最终输出定义生成器模型
    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model


# 判别器会把输入当成真实或虚假的图像，并输出一个分数。
def discriminator_model():
    """Build discriminator architecture."""
    # 3层网络，不使用sigmoid激活函数。
    n_layers, use_sigmoid = 3, False
    # 定义判别器的输入张量，shape为 ?*256*256*3
    inputs = Input(shape=input_shape_discriminator)
    # 卷积核个数64, 卷积方式same，输入图像大小减半，shape变化为：?*256*256*3 -> ?*128*128*64
    x = Conv2D(filters=ndf, kernel_size=(4, 4), strides=2, padding='same')(inputs)
    #  Leaky ReLU函数作为激活函数，给所有负值赋予一个非零斜率0.2，防止神经元的死亡。
    x = LeakyReLU(0.2)(x)

    # 3次卷积操作，每次卷积图像大小减半
    # shape变化过程：?*128*128*64->?*64*64*64->?*32*32*128->?*16*16*256
    nf_mult, nf_mult_prev = 1, 1   # nf_mult_prev有啥用？？？
    for n in range(n_layers):
        nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
        x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    # nf_mult = 8
    nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
    # 卷积操作，shape变化：?*16*16*256 -> ?*16*16*512
    x = Conv2D(filters=ndf*nf_mult, kernel_size=(4, 4), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # 卷积操作 shape：?*16*16*512 -> ？*16*16*1
    x = Conv2D(filters=1, kernel_size=(4, 4), strides=1, padding='same')(x)
    # use_sigmoid为false，不用sigmoid激活函数
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    # 将输出x展平，为全连接做准备
    x = Flatten()(x)
    # 定义一个全连接中间层，输出维度为1024，使用tanh作为激活函数
    x = Dense(1024, activation='tanh')(x)
    # 定义一个全连接输出层，输出维度为1(一个单一值)
    # 使用sigmoid作为激活函数，判别生成器生成图像的真假，输出结果限制在0到1之间
    x = Dense(1, activation='sigmoid')(x)

    # 使用inputs输入，x输出定义判别器模型
    model = Model(inputs=inputs, outputs=x, name='Discriminator')
    return model


def generator_containing_discriminator(generator, discriminator):
    inputs = Input(shape=image_shape)
    generated_image = generator(inputs)
    outputs = discriminator(generated_image)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# 创建完整的模型
def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    # 输入的图像张量
    inputs = Input(shape=image_shape)
    # 生成器输出
    generated_image = generator(inputs)
    # 判别器输出
    outputs = discriminator(generated_image)
    # 定义一个模型，将inputs作为模型输入，生成器输出结果和判别器输出结果连接起来作为模型的输出
    model = Model(inputs=inputs, outputs=[generated_image, outputs])
    return model


if __name__ == '__main__':
    g = generator_model()
    g.summary()
    d = discriminator_model()
    d.summary()
    m = generator_containing_discriminator(generator_model(), discriminator_model())
    m.summary()
