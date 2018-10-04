import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
AT=np.load('mat/AT.npy')
V=np.load('mat/V.npy')
AP=np.load('mat/AP.npy')
RH=np.load('mat/RH.npy')
PE=np.load('mat/PE.npy')
AT_train,AT_test=AT[0:8000],AT[8000:9568]
V_train,V_test=V[0:8000],V[8000:9568]
AP_train,AP_test=AP[0:8000],AP[8000:9568]
RH_train,RH_test=RH[0:8000],RH[8000:9568]
PE_train,PE_test=PE[0:8000],PE[8000:9568]
PE_train=PE_train.reshape(8000,1)
PE_test=PE_test.reshape(1568,1)

train,test=[],[]
for i in range(0,8000):
    temp=[]
    temp.append(AT_train[i])
    temp.append(V_train[i])
    temp.append(AP_train[i])
    temp.append(RH_train[i])
    temp=np.array(temp)
    train.append(temp)
train=np.array(train)
for i in range(0,1568):
    temp=[]
    temp.append(AT_test[i])
    temp.append(V_test[i])
    temp.append(AP_test[i])
    temp.append(RH_test[i])
    temp=np.array(temp)
    test.append(temp)
test=np.array(test)
X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None,1])
weights = {
    'h1': tf.Variable(tf.random_normal([4, 10])),#4 inputs 10  nodes in h1 layer
    'h2': tf.Variable(tf.random_normal([10, 10])),# 10 nodes in h2 layer
    'out': tf.Variable(tf.random_normal([10, 1]))# 1 ouput label
}
biases = {
    'b1': tf.Variable(tf.random_normal([10])),
    'b2': tf.Variable(tf.random_normal([10])),
    'out': tf.Variable(tf.random_normal([1]))
}
def neural_net(x):
    #hidden layer 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # output layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return (out_layer)
Y_hat=neural_net(X)
loss_op=tf.losses.huber_loss(Y,Y_hat)# huber loss advanced loss for regression
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss_op)
init = tf.global_variables_initializer()
epoch=2000
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    for i in range(0,epoch):
        sess.run(train_op,feed_dict={X:train,Y:PE_train})
        loss=sess.run(loss_op,feed_dict={X:train,Y:PE_train})
        if(i%100==0):
            print("epoch no"+str(i),(loss))
        pred=sess.run(Y_hat,feed_dict={X:test})
    plt.plot((pred), color='red', label='Prediction')
    plt.plot(PE_test, color='blue', label='Orignal')
    plt.legend(loc='upper left')
    plt.show()
    count=0
    for i in range(0,len(pred)):
        if(abs(pred[i]-PE_test[i])<0.5):
            count=count+1
    acc=100*(count/len(pred))
    print("accuracy",acc)

