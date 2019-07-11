import os
import datetime
import click
import numpy as np
import tqdm

from deblurgan.utils import load_images, write_log
from deblurgan.losses import wasserstein_loss, perceptual_loss
from deblurgan.model import generator_model, discriminator_model, generator_containing_discriminator_multiple_outputs

from keras.callbacks import TensorBoard
from keras.optimizers import Adam

BASE_DIR = 'weights/'


def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join(BASE_DIR, '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)


# 用生成器创建基于模糊图像的虚假输入，用真实和虚假这两种输入训练鉴别器，
# 最后训练将鉴别器和生成器连接在一起构成模型，其中鉴别器的权重是保持不变的。
# 将两个神经网络连接在一起的原因是生成器的输出一般没有反馈。
# 我们唯一的衡量指标是鉴别器是否接受生成的样本。

def train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    # 加载模糊图片和清晰图片数据
    data = load_images('./images/train', n_images)
    y_train, x_train = data['B'], data['A']

    # 初始化生成器模型
    g = generator_model()
    # 初始化判别器模型
    d = discriminator_model()
    # 初始化生成器和判别器连接成的完整模型
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    # 使用Adam最为判别器和完整模型的优化器
    # 优化器学习率为0.0001，一阶矩估计的指数衰减率为0.9，二阶矩估计的指数衰减率维0.999，
    # 模糊因子为1e-08，防止在实现中除以0
    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # 设置当前判别器可训练
    d.trainable = True
    # 编译判别器模型，使用d_opt最为优化器，使用wasserstein_loss作为损失函数
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    # 设置当前判别器模型不可训练
    d.trainable = False
    # 将感知损失和推土机损失结合起来使用，设置二者的比重为100:1。
    # 其中感知损失在生成器的输出上直接计算得来，感知损失能保证GAN模型面向去模糊任务
    # Wasserstein损失，在整个模型的输出上执行得来。它取自两张图像之间差距的平均值，优化GAN网络的收敛
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 1]
    # 编译完整模型，使用d_on_g_opt作为优化器，使用比重100:1的联合损失作为目标函数
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    # 设置当前的判别器模型可训练
    d.trainable = True

    #
    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

    # 存放日志的路径
    log_path = './logs'
    # tensorboard可视化
    tensorboard_callback = TensorBoard(log_path)

    # 根据两种损失依次训练鉴别器和生成器。我们用生成器来生成虚假输入，训练鉴别器从真实输入中分辨出虚假
    # 输入，然后训练整个模型。
    # 最外层迭代epoch_num次
    for epoch in tqdm.tqdm(range(epoch_num)):
        # 随机生成n_images大小的排列数，后续用于将训练集中的模糊图片和清晰图片随机划分批次
        permutated_indexes = np.random.permutation(x_train.shape[0])

        # 存放判别器模型损失
        d_losses = []
        # 存放完整模型损失
        d_on_g_losses = []
        # 每一个epoch 迭代n_images/batch_size次
        for index in range(int(x_train.shape[0] / batch_size)):
            # 每次迭代选出batch_size大小的模糊图片和清晰图片
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            # 存放每一轮迭代的batch_size张模糊图片
            image_blur_batch = x_train[batch_indexes]
            # 存放每一轮迭代的batch_size张清晰图片
            image_full_batch = y_train[batch_indexes]

            # 生成虚假输入
            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            # 每一轮迭代中，用本batch_size个虚假和真实输入训练判别器critic_updates次(默认5次)，
            # 5次训练中每次又会使用清晰图片（标签是1）和模糊图片（标签是0）分别对判别器做一次训练，
            # 相当于10次训练
            # 判别器loss函数使用的是wasserstein距离
            for _ in range(critic_updates):
                # d_loss_real 是当判别器预测图像为假但实际上为真时的损失
                # d_loss_fake 是当判别器预测图像为真但实际上为假时的损失。
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                # 判别器损失是d_loss_real和d_loss_fake的损失之和
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            # critic_updates次训练后，关闭判别器参数更新，使判别器不可训练，以下训练生成器，生成器的优化标准有两个，
            # 一个是和清晰图像的差异，一个是迷惑判别器的能力。
            d.trainable = False

            # 生成器和判别器的联合网络输出是生成器生成的图片+判别器的判别值（0到1），所以联合
            # 网络d_on_g的train_on_batch训练函数的第二个参数（该参数应该传入训练数据的真实标签）
            # 含有两个值，一个是真实清晰图像，一个是真实图像的标签（为1）
            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])

            # d_on_g损失函数优化的目标：
            # 生成器生成的图像和清晰图像的差异越来越小（感知损失，使用VGG16提取特征并比较），
            # 并且该生成图像经过判别器后的输出跟清晰图片经过判别器的输出的差异越来越小(使用wasserstein距离)
            # 两个loss的比重为100:1
            d_on_g_losses.append(d_on_g_loss)

            # 一轮循环过后，设置判别器可训练
            d.trainable = True

        # write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
        print(np.mean(d_losses), np.mean(d_on_g_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))


@click.command()
@click.option('--n_images', default=-1, help='Number of images to load for training')
@click.option('--batch_size', default=16, help='Size of batch')
@click.option('--log_dir', required=True, help='Path to the log_dir for Tensorboard')
@click.option('--epoch_num', default=4, help='Number of epochs for training')
@click.option('--critic_updates', default=5, help='Number of discriminator training')
def train_command(n_images, batch_size, log_dir, epoch_num, critic_updates):
    return train_multiple_outputs(n_images, batch_size, log_dir, epoch_num, critic_updates)


if __name__ == '__main__':
    train_command()
