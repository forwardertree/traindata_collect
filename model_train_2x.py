"""
신경망 훈련
"""

import tensorflow as tf
import numpy as np
import pickle
import glob

'''
----------------------------------------------------------------------------
csv 파일을 파이썬에 입력
----------------------------------------------------------------------------
'''
base_path = 'C:/Users/jjm/Desktop/ecg_ai/trainingDB/ecgdvq_62.5/'
file_list = [
    base_path + 'trainingDB_9_13_240404.pickle',
    base_path + 'trainingDB_9_13_240405_1001_p.pickle',
    base_path + 'trainingDB_9_13_240405_1006_p.pickle',
    base_path + 'trainingDB_9_13_240405_1007_p.pickle',
    base_path + 'trainingDB_9_13_240405_1102_p.pickle',
    base_path + 'trainingDB_9_13_240405_1103_p.pickle'
]

# 불러온 데이터를 저장할 리스트 또는 딕셔너리 초기화
loaded_data = []

trainingdb_aggregated = {}

for file_name in file_list:
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        for key in data:
            if key in trainingdb_aggregated:
                # numpy.ndarray인 경우 numpy.concatenate 사용
                trainingdb_aggregated[key] = np.concatenate((trainingdb_aggregated[key], data[key]))
            else:
                trainingdb_aggregated[key] = data[key]


# trainingdb_aggregated에서 훈련 데이터 설정
train_inp_ps = np.array(trainingdb_aggregated['inp_ps'], dtype='float32')
train_inp_pe = np.array(trainingdb_aggregated['inp_pe'], dtype='float32')
train_inp_qrss = np.array(trainingdb_aggregated['inp_qrss'], dtype='float32')
train_inp_qrse = np.array(trainingdb_aggregated['inp_qrse'], dtype='float32')
train_inp_ts = np.array(trainingdb_aggregated['inp_ts'], dtype='float32')
train_inp_te = np.array(trainingdb_aggregated['inp_te'], dtype='float32')
train_outp_ps = np.array(trainingdb_aggregated['ps'], dtype='float32')
train_outp_pe = np.array(trainingdb_aggregated['pe'], dtype='float32')
train_outp_qrss = np.array(trainingdb_aggregated['qrss'], dtype='float32')
train_outp_qrse = np.array(trainingdb_aggregated['qrse'], dtype='float32')
train_outp_ts = np.array(trainingdb_aggregated['ts'], dtype='float32')
train_outp_te = np.array(trainingdb_aggregated['te'], dtype='float32')

# with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_240404.pickle', 'rb') as f:
#     trainingdb = pickle.load(f)
#
# train_inp_ps = np.array(trainingdb['inp_ps'], dtype='float32')
# train_inp_pe = np.array(trainingdb['inp_pe'], dtype='float32')
# train_inp_qrss = np.array(trainingdb['inp_qrss'], dtype='float32')
# train_inp_qrse = np.array(trainingdb['inp_qrse'], dtype='float32')
# train_inp_ts = np.array(trainingdb['inp_ts'], dtype='float32')
# train_inp_te = np.array(trainingdb['inp_te'], dtype='float32')
# train_outp_ps = np.array(trainingdb['ps'], dtype='float32')
# train_outp_pe = np.array(trainingdb['pe'], dtype='float32')
# train_outp_qrss = np.array(trainingdb['qrss'], dtype='float32')
# train_outp_qrse = np.array(trainingdb['qrse'], dtype='float32')
# train_outp_ts = np.array(trainingdb['ts'], dtype='float32')
# train_outp_te = np.array(trainingdb['te'], dtype='float32')

dnn_model = {}
acc_cost_arr = np.array(['epoch', 'acc', 'cost'])

'''
----------------------------------------------------------------------------
pickle 파일로 저장된 학습모델 불러오기
----------------------------------------------------------------------------
'''
# save_path = 'K:/ecgrdvq_ai_model_220328.pickle'
# with open(save_path, 'rb') as f:
#     saved_dnn_model = pickle.load(f)

'''
----------------------------------------------------------------------------
신경망 파라미터 값 결정
----------------------------------------------------------------------------
'''
tf.reset_default_graph()
tf.set_random_seed(1237)

inp_col = 160
outp_col = 22
h1_col = 100
h2_col = 50
h3_col = 22

learning_rate = 0.001
train_num = 5000  # 훈련 횟수(= epoch)
# batch_size = 50
drop_out = 0.7

'''
----------------------------------------------------------------------------
심층신경망 Weight와 Bias 그리고 입출력 크기 결정 ps
----------------------------------------------------------------------------
'''
print('--------------------', 'Weight & Bias Setting', '--------------------', sep='\n', end='\n''\n')

with tf.name_scope("placeholder") as scope:
    x = tf.placeholder(tf.float32, [None, inp_col])
    y = tf.placeholder(tf.float32, [None, outp_col])
    keep_prob = tf.placeholder(tf.float32)  # 드랍아웃을 위한 변수 설정

with tf.name_scope("HL1") as scope:
    # w1 = saved_dnn_model['ps']['w1']
    # w1 = tf.convert_to_tensor(w1, dtype=tf.float32)
    # w1 = tf.Variable(w1)
    # b1 = saved_dnn_model['ps']['b1']
    # b1 = tf.convert_to_tensor(b1, dtype=tf.float32)
    # b1 = tf.Variable(b1)
    w1 = tf.get_variable("W1", shape=[inp_col, h1_col], dtype=tf.float32,
                         initializer=tf.keras.initializers.he_normal())
    b1 = tf.Variable(tf.random_normal([1, h1_col], stddev=0.01))
    z1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    z1 = tf.nn.dropout(z1, keep_prob=keep_prob)

with tf.name_scope("HL2") as scope:
    # w2 = saved_dnn_model['ps']['w2']
    # w2 = tf.convert_to_tensor(w2, dtype=tf.float32)
    # w2 = tf.Variable(w2)
    # b2 = saved_dnn_model['ps']['b2']
    # b2 = tf.convert_to_tensor(b2, dtype=tf.float32)
    # b2 = tf.Variable(b2)
    w2 = tf.get_variable("W2", shape=[h1_col, h2_col], dtype=tf.float32,
                         initializer=tf.keras.initializers.he_normal())
    b2 = tf.Variable(tf.random_normal([1, h2_col], stddev=0.01))
    z2 = tf.nn.relu(tf.matmul(z1, w2) + b2)
    z2 = tf.nn.dropout(z2, keep_prob=keep_prob)

with tf.name_scope("OL") as scope:
    # w3 = saved_dnn_model['ps']['w3']
    # w3 = tf.convert_to_tensor(w3, dtype=tf.float32)
    # w3 = tf.Variable(w3)
    # b3 = saved_dnn_model['ps']['b3']
    # b3 = tf.convert_to_tensor(b3, dtype=tf.float32)
    # b3 = tf.Variable(b3)
    w3 = tf.get_variable("W3", shape=[h2_col, h3_col], dtype=tf.float32,
                         initializer=tf.keras.initializers.he_normal())
    b3 = tf.Variable(tf.random_normal([1, h3_col], stddev=0.01))
    z = tf.matmul(z2, w3) + b3

'''
----------------------------------------------------------------------------
훈련 및 정확도 조건 설정
----------------------------------------------------------------------------
'''
print('--------------------', 'Train & Acc Setting', '--------------------', sep='\n', end='\n''\n')

with tf.name_scope("train") as scope:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y))
    op_train = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    pred = tf.nn.softmax(z)
    prediction = tf.argmax(pred, 1)
    true_Y = tf.argmax(y, 1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, true_Y), dtype=tf.float32))

'''
----------------------------------------------------------------------------
텐서플로우 Session 시작 ps
----------------------------------------------------------------------------
'''
tmp = np.array(['epoch_ps', 'acc_ps', 'cost_ps'])
acc_cost_arr = np.vstack((acc_cost_arr, tmp))

# ps 학습 시작
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_num):
        sess.run(op_train, feed_dict={x: train_inp_ps, y: train_outp_ps, keep_prob: drop_out})

        acc_train = sess.run(accuracy, feed_dict={x: train_inp_ps, y: train_outp_ps, keep_prob: 1.0})
        cost_train = sess.run(cost, feed_dict={x: train_inp_ps, y: train_outp_ps, keep_prob: 1.0})

        tmp_arr = np.array([epoch, acc_train, cost_train])
        acc_cost_arr = np.vstack((acc_cost_arr, tmp_arr))

        if epoch % 1 == 0:
            print("ps, epoch: %d, train acc: %.2f, train cost: %.2f" % (epoch, acc_train, cost_train), end='\r')

    # Weight 및 Bias 저장
    tmp_dict = {}
    tmp_dict['w1'] = sess.run(w1)
    tmp_dict['w2'] = sess.run(w2)
    tmp_dict['w3'] = sess.run(w3)

    tmp_dict['b1'] = sess.run(b1)
    tmp_dict['b2'] = sess.run(b2)
    tmp_dict['b3'] = sess.run(b3)

    dnn_model['ps'] = tmp_dict

'''
----------------------------------------------------------------------------
텐서플로우 Session 시작 pe
----------------------------------------------------------------------------
'''
tmp = np.array(['epoch_pe', 'acc_pe', 'cost_pe'])
acc_cost_arr = np.vstack((acc_cost_arr, tmp))

# pe 학습 시작
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_num):
        sess.run(op_train, feed_dict={x: train_inp_pe, y: train_outp_pe, keep_prob: drop_out})

        acc_train = sess.run(accuracy, feed_dict={x: train_inp_pe, y: train_outp_pe, keep_prob: 1.0})
        cost_train = sess.run(cost, feed_dict={x: train_inp_pe, y: train_outp_pe, keep_prob: 1.0})

        tmp_arr = np.array([epoch, acc_train, cost_train])
        acc_cost_arr = np.vstack((acc_cost_arr, tmp_arr))

        if epoch % 1 == 0:
            print("pe, epoch: %d, train acc: %.2f, train cost: %.2f" % (epoch, acc_train, cost_train), end='\r')

    # Weight 및 Bias 저장
    tmp_dict = {}
    tmp_dict['w4'] = sess.run(w1)
    tmp_dict['w5'] = sess.run(w2)
    tmp_dict['w6'] = sess.run(w3)

    tmp_dict['b4'] = sess.run(b1)
    tmp_dict['b5'] = sess.run(b2)
    tmp_dict['b6'] = sess.run(b3)

    dnn_model['pe'] = tmp_dict

'''
----------------------------------------------------------------------------
텐서플로우 Session 시작 qrss
----------------------------------------------------------------------------
'''
tmp = np.array(['epoch_qrss', 'acc_qrss', 'cost_qrss'])
acc_cost_arr = np.vstack((acc_cost_arr, tmp))

# qrss 학습 시작
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_num):
        sess.run(op_train, feed_dict={x: train_inp_qrss, y: train_outp_qrss, keep_prob: drop_out})

        acc_train = sess.run(accuracy, feed_dict={x: train_inp_qrss, y: train_outp_qrss, keep_prob: 1.0})
        cost_train = sess.run(cost, feed_dict={x: train_inp_qrss, y: train_outp_qrss, keep_prob: 1.0})

        tmp_arr = np.array([epoch, acc_train, cost_train])
        acc_cost_arr = np.vstack((acc_cost_arr, tmp_arr))

        if epoch % 1 == 0:
            print("qrss, epoch: %d, train acc: %.2f, train cost: %.2f" % (epoch, acc_train, cost_train), end='\r')

    # Weight 및 Bias 저장
    tmp_dict = {}
    tmp_dict['w7'] = sess.run(w1)
    tmp_dict['w8'] = sess.run(w2)
    tmp_dict['w9'] = sess.run(w3)

    tmp_dict['b7'] = sess.run(b1)
    tmp_dict['b8'] = sess.run(b2)
    tmp_dict['b9'] = sess.run(b3)

    dnn_model['qrss'] = tmp_dict

'''
----------------------------------------------------------------------------
텐서플로우 Session 시작 qrse
----------------------------------------------------------------------------
'''
tmp = np.array(['epoch_qrse', 'acc_qrse', 'cost_qrse'])
acc_cost_arr = np.vstack((acc_cost_arr, tmp))

# qrse 학습 시작
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_num):
        sess.run(op_train, feed_dict={x: train_inp_qrse, y: train_outp_qrse, keep_prob: drop_out})

        acc_train = sess.run(accuracy, feed_dict={x: train_inp_qrse, y: train_outp_qrse, keep_prob: 1.0})
        cost_train = sess.run(cost, feed_dict={x: train_inp_qrse, y: train_outp_qrse, keep_prob: 1.0})

        tmp_arr = np.array([epoch, acc_train, cost_train])
        acc_cost_arr = np.vstack((acc_cost_arr, tmp_arr))

        if epoch % 1 == 0:
            print("qrse, epoch: %d, train acc: %.2f, train cost: %.2f" % (epoch, acc_train, cost_train), end='\r')

    # Weight 및 Bias 저장
    tmp_dict = {}
    tmp_dict['w10'] = sess.run(w1)
    tmp_dict['w11'] = sess.run(w2)
    tmp_dict['w12'] = sess.run(w3)

    tmp_dict['b10'] = sess.run(b1)
    tmp_dict['b11'] = sess.run(b2)
    tmp_dict['b12'] = sess.run(b3)

    dnn_model['qrse'] = tmp_dict

'''
----------------------------------------------------------------------------
텐서플로우 Session 시작 ts
----------------------------------------------------------------------------
'''
tmp = np.array(['epoch_ts', 'acc_ts', 'cost_ts'])
acc_cost_arr = np.vstack((acc_cost_arr, tmp))

# ts 학습 시작
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_num):
        sess.run(op_train, feed_dict={x: train_inp_ts, y: train_outp_ts, keep_prob: drop_out})

        acc_train = sess.run(accuracy, feed_dict={x: train_inp_ts, y: train_outp_ts, keep_prob: 1.0})
        cost_train = sess.run(cost, feed_dict={x: train_inp_ts, y: train_outp_ts, keep_prob: 1.0})

        tmp_arr = np.array([epoch, acc_train, cost_train])
        acc_cost_arr = np.vstack((acc_cost_arr, tmp_arr))

        if epoch % 1 == 0:
            print("ts, epoch: %d, train acc: %.2f, train cost: %.2f" % (epoch, acc_train, cost_train), end='\r')

    # Weight 및 Bias 저장
    tmp_dict = {}
    tmp_dict['w13'] = sess.run(w1)
    tmp_dict['w14'] = sess.run(w2)
    tmp_dict['w15'] = sess.run(w3)

    tmp_dict['b13'] = sess.run(b1)
    tmp_dict['b14'] = sess.run(b2)
    tmp_dict['b15'] = sess.run(b3)

    dnn_model['ts'] = tmp_dict

'''
----------------------------------------------------------------------------
텐서플로우 Session 시작 te
----------------------------------------------------------------------------
'''
tmp = np.array(['epoch_te', 'acc_te', 'cost_te'])
acc_cost_arr = np.vstack((acc_cost_arr, tmp))

# te 학습 시작
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_num):
        sess.run(op_train, feed_dict={x: train_inp_te, y: train_outp_te, keep_prob: drop_out})

        acc_train = sess.run(accuracy, feed_dict={x: train_inp_te, y: train_outp_te, keep_prob: 1.0})
        cost_train = sess.run(cost, feed_dict={x: train_inp_te, y: train_outp_te, keep_prob: 1.0})

        tmp_arr = np.array([epoch, acc_train, cost_train])
        acc_cost_arr = np.vstack((acc_cost_arr, tmp_arr))

        if epoch % 1 == 0:
            print("te, epoch: %d, train acc: %.2f, train cost: %.2f" % (epoch, acc_train, cost_train), end='\r')

    # Weight 및 Bias 저장
    tmp_dict = {}
    tmp_dict['w16'] = sess.run(w1)
    tmp_dict['w17'] = sess.run(w2)
    tmp_dict['w18'] = sess.run(w3)

    tmp_dict['b16'] = sess.run(b1)
    tmp_dict['b17'] = sess.run(b2)
    tmp_dict['b18'] = sess.run(b3)

    dnn_model['te'] = tmp_dict

'''
pickle 파일로 저장
'''
save_path = r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\ecgrdvq_ai_model_240405_plus.pickle'
with open(save_path, 'wb') as f:
    pickle.dump(dnn_model, f)

'''
acc, cost 결과 저장
'''
np.savetxt(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\dnn_result_acc_240405_plus.csv', acc_cost_arr, fmt='%s', delimiter=",")
