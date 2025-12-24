
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
def variable_init(size):
    in_dim = size[0]
    #计算随机生成变量所服从的正态分布标准差
    w_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=w_stddev)
def  sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
#G网络
def generator(z, y,theta_G):
    inputs = tf.concat(axis=1, values=[z, y])
    #第一层先计算 y=z*G_W1+G-b1,然后投入激活函数计算G_h1=ReLU（y）,G_h1 为第二次层神经网络的输出激活值
    G_h1 = tf.nn.relu(tf.matmul(inputs, theta_G[0]) + theta_G[2])
    #以下两个语句计算第二层传播到第三层的激活结果，第三层的激活结果是含有784个元素的向量，该向量转化28×28就可以表示图像
    G_log_prob = tf.matmul(G_h1, theta_G[1]) + theta_G[3]
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob
#D网络，这里是一个简单的神经网络，x是输入图片向量，y是相应的label
def discriminator(x, y,theta_D):
    inputs = tf.concat(axis=1, values=[x, y])
    #计算D_h1=ReLU（x*D_W1+D_b1）,该层的输入为含784个元素的向量
    D_h1 = tf.nn.relu(tf.matmul(inputs, theta_D[0]) + theta_D[2])
    #计算第三层的输出结果。因为使用的是Sigmoid函数，则该输出结果是一个取值为[0,1]间的标量
    #即判别输入的图像到底是真（=1）还是假（=0）
    D_logit = tf.matmul(D_h1, theta_D[1]) + theta_D[3]
    D_prob = tf.nn.sigmoid(D_logit)
    #返回判别为真的概率和第三层的输入值，输出D_logit是为了将其输入tf.nn.sigmoid_cross_entropy_with_logits()以构建损失函数
    return D_prob, D_logit
#该函数用于输出生成图片
def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig

def cgan():
    #数据输入
    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    X_dim = mnist.train.images.shape[1]
    y_dim = mnist.train.labels.shape[1]
    mb_size = 64 #batchsize
    Z_dim = 100 #噪声
    h_dim = 128
    #X代表输入图片，28*28，y是相应的label
    X = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, y_dim])
    #权重，CGAN的输入是将图片输入与label concat起来，所以权重维度为784+10
    D_W1 = tf.Variable(variable_init([X_dim + y_dim, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    #第二层有h_dim个节点
    D_W2 = tf.Variable(variable_init([h_dim, 1]))
    D_b2 = tf.Variable(tf.zeros(shape=[1]))
    theta_D = [D_W1, D_W2, D_b1, D_b2]

    #G网络参数，输入维度为Z_dim+y_dim，中间层有h_dim个节点，输出X_dim的数据
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
    #权重
    G_W1 = tf.Variable(variable_init([Z_dim + y_dim, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    G_W2 = tf.Variable(variable_init([h_dim, X_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))
    
    theta_G = [G_W1, G_W2, G_b1, G_b2]

    #生成网络，基本和GAN一致
    G_sample = generator(Z, y,theta_G)
    D_real, D_logit_real = discriminator(X, y,theta_D)
    D_fake, D_logit_fake = discriminator(G_sample, y,theta_D)
    #优化式
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    #训练
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
    with tf.Session() as session:
        #初始化所有定义的变量
        session.run(tf.global_variables_initializer())

        #如果当前目录下不存在out文件夹，则创建该文件夹
        if not os.path.exists('out/'):
            os.makedirs('out/')
        #初始化，并开始迭代训练，10W次
        i = 0
        for it in range(100001):
            if it % 2000 == 0:
                #n_sample 是G网络测试用的Batchsize，为16，所以输出的png图有16张
                n_sample = 16
        
                Z_sample = sample_Z(n_sample, Z_dim)#输入的噪声，尺寸为batchsize*noise维度
                y_sample = np.zeros(shape=[n_sample, y_dim])#输入的label，尺寸为batchsize*label维度
                y_sample[:, 4] = 1 #输出4
        
                samples = session.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})#G网络的输入
        
                fig = plot(samples)
                plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')#输出生成的图片
                i += 1
                plt.close(fig)
            #mb_size是网络训练时用的Batchsize，为128
            X_mb, y_mb = mnist.train.next_batch(mb_size)
            #Z_dim是noise的维度，为100
            Z_sample = sample_Z(mb_size, Z_dim)
            #交替最小化训练
            _, D_loss_curr = session.run([D_solver, D_loss], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
            _, G_loss_curr = session.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})
            #输出训练时的参数
            if it % 2000 == 0:
                print('Iter: {}'.format(it))
                print('D loss: {:.4}'. format(D_loss_curr))
                print('G_loss: {:.4}'.format(G_loss_curr))
                print()
if __name__=='__main__':
    cgan()