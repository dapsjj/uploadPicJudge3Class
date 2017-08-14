import os
import numpy as np
import tensorflow as tf
import input_data
import model
import datetime

# %%

N_CLASSES = 3
IMG_W = 208  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 208
BATCH_SIZE = 20
CAPACITY = 500
MAX_STEP = 1500  # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001  # with current parameters, it is suggested to use learning rate<0.0001


def run_training():

    # you need to change the directories to yours.
    # train_dir = '/home/kevin/tensorflow/hams_vs_hots/data/train/'
    train_dir = 'D:/workspace/uploadPicJudge3Class/train/'
    # logs_train_dir = '/home/kevin/tensorflow/hams_vs_hots/logs/train/'
    logs_train_dir = 'D:/workspace/uploadPicJudge3Class/logs/'

    train, train_label = input_data.get_files(train_dir)

    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE,
                                                          CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])

            if step % 10 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0)+'  '+datetime.datetime.now().strftime('%Y-%m-%d %H_%M_%S'))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 100 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# %% Evaluate one image
# when training, comment the following codes.


# from PIL import Image
# import matplotlib.pyplot as plt
#
#
def get_one_image(img):
    image = np.array(img)
    return image
#
#
def evaluate_one_image(pic):
    image_array = get_one_image(pic)
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3

        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)

        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # you need to change the directories to yours.
        logs_train_dir = 'D:/workspace/uploadPicJudge3Class/logs/'

        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index == 0:
                # print('This is a hamberger with possibility %.6f' % prediction[:, 0])
                msg = 'This is a bread with possibility %.4f%%' % (prediction[:, 0] * 100)
                return msg

            elif max_index == 1:
                # print('This is a hamberger with possibility %.6f' % prediction[:, 1])
                msg = 'This is a hotdog with possibility %.4f%%' % (prediction[:, 1] * 100)
                return msg
            else:
                # print('This is a hotdog with possibility %.6f' % prediction[:, 2])
                msg = 'This is a hamberger with possibility %.4f%%' % (prediction[:, 2] * 100)
                return msg


# %%

# run_training()

# evaluate_one_image()


