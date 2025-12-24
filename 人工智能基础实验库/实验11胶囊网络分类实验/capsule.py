import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
class CapsLayer(object):
    def __init__(self, epsilon,iter_routing,num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.epsilon = epsilon
        self.iter_routing = iter_routing
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        #初级胶囊层 
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride
            if not self.with_routing:
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,self.kernel_size, self.stride, padding="VALID")
                capsules = tf.reshape(capsules, (-1, capsules.shape[1].value*capsules.shape[2].value*self.num_outputs, self.vec_len, 1))
                capsules = self.squash(capsules)
                return (capsules)
        #初级胶囊转化为高级胶囊
        if self.layer_type == 'FC':
            if self.with_routing:
                self.input = tf.reshape(input, shape=(-1, input.shape[1].value, 1, input.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    #初始化b_IJ为0
                    b_IJ = tf.constant(np.zeros([1, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = self.routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)
            return(capsules)
    #动态路由
    def routing(self,input, b_IJ):
        input_shape = input.get_shape()
        W = tf.get_variable('Weight', shape=(1,input_shape[1],self.num_outputs*self.vec_len,input_shape[3],input_shape[4]),dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
        input = tf.tile(input, [1, 1,self.num_outputs* self.vec_len, 1, 1])
        u_hat = tf.reduce_sum(tf.multiply(W,input), axis=3, keepdims=True)
        u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], self.num_outputs, self.vec_len, 1])
        u_hat_stopped = tf.stop_gradient(u_hat,name='stop_gradient')
        for r_iter in range(self.iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                c_IJ = tf.nn.softmax(b_IJ, axis=2)
                if r_iter == self.iter_routing-1:
                    s_J = tf.multiply(c_IJ, u_hat)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)
                elif r_iter <self.iter_routing-1:
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)
                    v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                    u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                    b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
        return(v_J)
    # squash激活函数
    def squash(self,vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + self.epsilon)
        vec_squashed = scalar_factor * vector  
        return(vec_squashed)


class Capsnet():
    def __init__(self,image_size,num_classes,lambda_val,m_plus,m_minus,epsilon,iter_routing,num_outputs_decode,num_dims_decode):
        self.image_size = image_size
        self.lambda_val = lambda_val
        # 设定上界
        self.m_plus = m_plus
        # 设定下界
        self.m_minus = m_minus
        self.epsilon = epsilon
        self.iter_routing = iter_routing
        # 设定第一个卷积层输出个数
        self.num_outputs_layer_conv2d = 256
        # 设定初级胶囊输出个数
        self.num_outputs_layer_PrimaryCaps = 32
        # 设定初级胶囊输出的向量大小
        self.num_dims_layer_PrimaryCaps = 8
        # 设定高级胶囊输出个数
        self.num_outputs_decode = num_outputs_decode
        # 设定高级胶囊输出的向量大小
        self.num_dims_decode = num_dims_decode
        # 设定初始输入图片为28X28
        self.image = tf.placeholder(tf.float32,[None,image_size,image_size,1])
        self.label = tf.placeholder(tf.int64,[None,1])
        label_onehot = tf.one_hot(self.label,depth=num_classes,axis=1,dtype = tf.float32)
        # 卷积层
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(self.image, num_outputs = self.num_outputs_layer_conv2d,kernel_size=9, stride=1,padding='VALID')
        # 初级胶囊层
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(self.epsilon,self.iter_routing,num_outputs=self.num_outputs_layer_PrimaryCaps, vec_len=self.num_dims_layer_PrimaryCaps, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=9, stride=2)
        # 高级胶囊
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(self.epsilon,self.iter_routing, num_outputs = self.num_outputs_decode, vec_len=self.num_dims_decode, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)
        # 计算误差和重构
        with tf.variable_scope('Masking'):
            mask_with_y=True
            if mask_with_y:
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + self.epsilon)
                self.masked_v = tf.matmul(tf.squeeze(self.caps2,axis = 3), tf.reshape(label_onehot, (-1, self.num_outputs_decode, 1)), transpose_a=True)
        # 重构图像
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(-1, self.num_dims_decode))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)
        # 分类损失
        max_l = tf.square(tf.maximum(0., self.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - self.m_minus))
        max_l_out = tf.reshape(max_l, shape=(-1,self.num_outputs_decode))
        max_r_out = tf.reshape(max_r, shape=(-1,self.num_outputs_decode))
        label_onehot_out = tf.squeeze(label_onehot,axis=2)
        L_c = label_onehot_out * max_l_out + self.lambda_val * (1 - label_onehot_out) * max_r_out
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        # 重构损失
        orgin = tf.reshape(self.image, shape=(-1, self.image_size*self.image_size))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)
        self.total_loss = self.margin_loss + 0.0005 * self.reconstruction_err
        # Adam optimization
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.train.AdamOptimizer().minimize(self.total_loss, global_step=self.global_step)

class run_main():
    def __init__(self):
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
        self.image_size = 28
        self.num_classes = 10
        self.batch_size = 32
        self.lambda_val = 0.5
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.epsilon = 1e-9
        self.iter_routing = 3
        self.num_outputs_decode = 10
        self.num_dims_decode = 16
        self.capsnet_model = Capsnet(self.image_size,self.num_classes,self.lambda_val,self.m_plus,self.m_minus,self.epsilon,self.iter_routing,self.num_outputs_decode,self.num_dims_decode)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self,iteration):
        for step in tqdm(range(iteration), total=iteration, ncols=70, leave=False, unit='b'):
            image, labels = self.mnist.train.next_batch(self.batch_size)
            image = image.reshape((self.batch_size,self.image_size,self.image_size,1))
            labels = labels.reshape((-1,1))
            output_feed = [self.capsnet_model.train_op, self.capsnet_model.total_loss]
            _, loss= self.sess.run(output_feed, feed_dict={self.capsnet_model.image: image, self.capsnet_model.label: labels})
            if step % 1 == 0:
                print('step {}: loss = {:3.4f}'.format(step, loss))
            
    def predict(self):
        for dataset in [self.mnist.test]:
            steps_per_epoch = dataset.num_examples // self.batch_size
            correct = 0
            num_samples = steps_per_epoch * self.batch_size
            for test_step in range(steps_per_epoch):
                image, labels = self.mnist.train.next_batch(self.batch_size)
                image = image.reshape((self.batch_size,self.image_size,self.image_size,1))
                v_length = self.sess.run(self.capsnet_model.v_length,feed_dict={self.capsnet_model.image: image})
                prediction = np.argmax(np.squeeze(v_length),axis=1) 
                correct += np.sum(prediction == labels)
            acc = correct / num_samples
            print('test accuracy = {}'.format(acc))

if __name__ == "__main__":
    ram_better = run_main()
    ram_better.train(200) 
    ram_better.predict()