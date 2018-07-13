import time
start=time.time()
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
from batch import*



n_inputs=4
n_neurons=10
n_steps=30 #no of days here
n_outputs=4
batch_size=1



window_size=30

x=tf.placeholder(shape=[None,n_steps, n_inputs],dtype=tf.float32)
y=tf.placeholder(shape=[None,n_steps, n_outputs],dtype=tf.float32)
cell = tf.contrib.rnn.OutputProjectionWrapper(
tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.relu),
output_size=n_outputs)
outputs,states=tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)


learning_rate=0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

n_iterations = 4674
#batch_size = 24
tot_epoches=10
min_val=1000
min_num=1000


with tf.Session() as sess:
    init.run()
    mse_train_total=[];mse_valid_total=[]   
    for epoch in range(tot_epoches):
        for iteration in range(n_iterations):
            if iteration % 5 != 0:
                x_batch, y_batch = next_batch_train(window_size,iteration+1, batch_size, n_inputs, n_outputs)		
                sess.run(training_op, feed_dict={x: x_batch , y: y_batch})
                mse_train = loss.eval(feed_dict={x: x_batch, y: y_batch})
                print(int(n_iterations*epoch+iteration), "\tMSE_train:", mse_train)
                mse_train_total.append([n_iterations*epoch+iteration , mse_train])
            if iteration % 5 == 0:
                x_val,y_val=next_batch_train(window_size,iteration+1, batch_size, n_inputs, n_outputs)
                mse_valid= loss.eval(feed_dict={x: x_val, y: y_val})
                print(int(n_iterations*epoch+iteration), "\tMSE_valid:",mse_valid)
                mse_valid_total.append([n_iterations*epoch+iteration , mse_valid])
    #y_test_com=np.array([])
    mse_train_data=pd.DataFrame(mse_train_total)
    mse_train_data.to_csv('mse_train_file',sep='\t')
    mse_valid_data=pd.DataFrame(mse_valid_total)
    mse_valid_data.to_csv('mse_valid_file',sep='\t')    
    n_test_samples=24        
    for i in range(n_test_samples):       
        x_test, y_test = next_batch_test(window_size,(30*i)+1, batch_size, n_inputs, n_outputs)      
        y_test_pred,y_test_loss=sess.run([outputs,loss],feed_dict={x:x_test, y:y_test})
        
        if i == 0:
            y_test_com=(y_test_pred*df8.astype(int).std(axis=0))+df8.mean(axis=0)
        else:
            y_test_com=np.concatenate((y_test_com,(y_test_pred*df8.astype(int).std(axis=0))+df8.mean(axis=0)), axis=1)        
        
    y_test_com=y_test_com.reshape(720,4)
    a=pd.DataFrame(y_test_com)
    a.to_csv('out', sep='\t')	
print("time",time.time()-start)


            
            
            
                      
            



			

