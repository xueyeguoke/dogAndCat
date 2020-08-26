import os
import numpy as np
import tensorflow as tf
import cv2 as cv
import time

img_w = 128
img_h = 128
img_c = 5
img_dir = 'E:/AI/AI14/tensorflow14/ddt_img_try/datas/train_reshape'
test_dir = 'E:/AI/AI14/tensorflow14/ddt_img_try/datas/test/test'
save_path = 'E:/AI/AI14/tensorflow14/ddt_img_try/samples'


def create_model(input_x):
    # 定义一个网络结构: input -> conv1 -> relu -> pooling -> conv2 -> relu -> pooling -> conv3 -> relu -> pooling -> FC -> relu -> FC
    with tf.variable_scope("net"):
        with tf.variable_scope('conv1'):
            filter = tf.get_variable(name='w', shape=[3, 3, img_c, 10])
            net = tf.nn.conv2d(input=input_x, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(value=net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('conv2'):
            filter = tf.get_variable(name='w', shape=[3, 3, 10, 20])
            net = tf.nn.conv2d(input=net, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(value=net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('conv3'):
            filter = tf.get_variable(name='w', shape=[3, 3, 20, 40])
            net = tf.nn.conv2d(input=net, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.relu(net)
            net = tf.nn.max_pool(value=net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('fc1'):
            shape = net.get_shape()
            dim_size = shape[1] * shape[2] * shape[3]
            net = tf.reshape(net, shape=[-1, dim_size])
            w = tf.get_variable(name='w', shape=[dim_size, 80])
            b = tf.get_variable(name='b', shape=[80])
            net = tf.add(tf.matmul(net, w), b)
            net = tf.nn.relu(net)

        with tf.variable_scope('fc2'):
            w = tf.get_variable(name='w', shape=[80, 2])
            b = tf.get_variable(name='b', shape=[2])
            logits = tf.add(tf.matmul(net, w), b)

        with tf.variable_scope("Prediction"):
            # 每行的最大值对应的下标就是当前样本的预测值
            predictions = tf.argmax(logits, axis=1)

        return logits, predictions


def create_loss(labels, logits):
    with tf.name_scope("loss"):
        # loss = tf.reduce_mean(-tf.log(tf.reduce_sum(labels * tf.nn.softmax(logits))))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        tf.summary.scalar('loss', loss)
    return loss


def create_train_op(learning_rate, loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    op = optimizer.minimize(loss=loss)
    return op


def create_accuracy(labels, predictions):
    with tf.name_scope("accuracy"):
        # 获取实际的类别下标，形状为[n_samples,]
        y_labels = tf.argmax(labels, 1)
        # 计算准确率
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_labels, predictions), tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    return accuracy


def train():
    with tf.Graph().as_default():
        input_x = tf.placeholder(name='input_x', shape=[None, img_w, img_h, img_c], dtype=tf.float32)
        input_y = tf.placeholder(name='input_y', shape=[None, 2], dtype=tf.float32)
        global_step = tf.train.get_or_create_global_step()

        logits, predict = create_model(input_x)
        loss = create_loss(input_y, logits)
        train_op = create_train_op(0.001, loss=loss)
        accuracy = create_accuracy(input_y, predict)
        batch_size = 100

        with tf.Session() as sess:
            var_list = tf.trainable_variables()
            var_list.append(global_step)
            saver = tf.train.Saver(var_list=var_list)

            sess.run(tf.global_variables_initializer())

            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("进行模型恢复操作...")
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

            img_paths = os.listdir(img_dir)
            random_index = np.random.permutation(len(img_paths))
            range_idx = 0
            while True:
                if range_idx >= len(img_paths) / batch_size:
                    range_idx = 0
                batch_idx = random_index[range_idx * batch_size:range_idx * batch_size + batch_size]
                batch_img_paths = np.array(img_paths)[batch_idx]
                x_train = []
                y_train = []
                for img_path in batch_img_paths:
                    if img_path.startswith('cat'):
                        y_train.append([1, 0])
                    elif img_path.startswith('dog'):
                        y_train.append([0, 1])
                    else:
                        pass
                    x_train.append(np.load(os.path.join(img_dir, img_path)))

                feed_dict = {
                    input_x: x_train,
                    input_y: y_train
                }
                _, loss_, accuracy_, global_step_ = sess.run([train_op, loss, accuracy, global_step],
                                                             feed_dict=feed_dict)

                if global_step_ % 3 == 0:
                    print('{}--global_step:{},loss:{},accuracy:{}'
                          .format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                  global_step_, loss_, accuracy_))

                if global_step_ % 100 == 0 and global_step_ > 1:
                    print('*' * 100)
                    # 持久化
                    # save_file = os.path.join(save_path, '{}_model_{}%.ckpt'.format(
                    #     time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())), accuracy_
                    # ))
                    save_file = os.path.join(save_path, 'model.ckpt')
                    print('保存模型', save_file)
                    saver.save(sess, save_path=save_file, global_step=global_step_)
                    # predict_ = sess.run([predict], feed_dict=feed_dict)
                    # print('y_train:', y_train, 'predict:', predict_)
                range_idx += 1
                global_step += 1


def test(test_dir):
    os.makedirs(os.path.join(test_dir, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(test_dir, 'dog'), exist_ok=True)
    with tf.Graph().as_default():
        input_x = tf.placeholder(name='input_x', shape=[None, img_w, img_h, img_c], dtype=tf.float32)
        global_step = tf.train.get_or_create_global_step()
        logits, predict = create_model(input_x)

        with tf.Session() as sess:
            var_list = tf.trainable_variables()
            var_list.append(global_step)
            saver = tf.train.Saver(var_list=var_list)
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(save_path)
            if ckpt and ckpt.model_checkpoint_path:
                print("进行模型恢复操作...")
                saver.restore(sess, ckpt.model_checkpoint_path)
                saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)

                files = os.listdir(test_dir)
                step = 0
                for file_name in files:
                    if not os.path.isfile(os.path.join(test_dir, file_name)):
                        continue
                    print('图片{}:{},共{}张图片'.format(step, os.path.join(test_dir, file_name), len(files)))
                    img = cv.imread(os.path.join(test_dir, file_name))
                    img_arr = pretreat_imges(img)  # 预处理图片
                    feed_dict = {input_x: [img_arr]}
                    predict_ = sess.run(predict, feed_dict=feed_dict)
                    if predict_[0] == 0:
                        cv.imwrite(os.path.join(test_dir, 'cat', file_name), img)
                    else:
                        cv.imwrite(os.path.join(test_dir, 'dog', file_name), img)
                    step += 1

    pass


def pretreat_imges(img):
    img = cv.resize(img, (128, 128))
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯模糊
    img_gaussiian = cv.GaussianBlur(img_gray, ksize=(5, 5), sigmaX=1, sigmaY=1)
    kernel = np.asarray([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    sobely = cv.filter2D(img_gaussiian, ddepth=6, kernel=kernel)
    sobelx = cv.filter2D(img_gaussiian, ddepth=6, kernel=kernel.T)
    sobely = (sobely + 256) / 512
    r, g, b = cv.split(img)
    sobely = np.uint8(sobely)
    sobelx = np.uint8(sobelx)
    img_arr = cv.merge((r, g, b, sobelx, sobely))
    return img_arr


def main():
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    train()
    # test(test_dir)


if __name__ == '__main__':
    # img = cv.imread('E:/AI/AI14/tensorflow14/ddt_img_try/datas/test/test/1.jpg')
    # cv.imshow('img', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    main()
