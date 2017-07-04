import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import sys
from TensorGenerator import TensorGenerator
from UnshuffledDataGenerator import UnshuffledDataGenerator


#STEP ONE: READ IN DATA

#Gives me two numpy arrays. One of them is NumDataPoints by 2 array (each row has an x and y coordinate) and the other is NumDataPoints by 1 array that just contains a bunch of labels.
ToGenerateData=TensorGenerator()

#train_x is the training coords, train_y is the training labels, test_x is the testing coords, test_y is the training labels. 
train_x,train_y,test_x,test_y,NumOfDataPoints=ToGenerateData.OutputData(sys.argv[1],sys.argv[2])

#Getting the Unshuffled Data passed in
ToGenerateUnshuffledData=UnshuffledDataGenerator()
Circle_2000,Circle_2000_Labels,Circle_8000,Circle_8000_Labels,Torus_2000,Torus_2000_Labels,Torus_8000,Torus_8000_Labels=ToGenerateUnshuffledData.OutputData()


#STEP TWO: SET UP THE CONSTANTS OF THE NN

#One hidden layer only
num_Nodes=500

#Number of output nodes
n_classes=2

#Batch Size
batch_size=100


#x is input (the x-coordinate and y-coordinate). y is labels. These are placeholders I will feed things through later on (Will feed the data to be analyzed through x and the labels through y)
x=tf.placeholder('float')
y=tf.placeholder('float')



#STEP THREE: SETTING UP THE ACTUAL LAYER(S) OF THE ANN.

#Setting up the computation graph of the ANN. The ANN has two inputs (the x-coordinate and the y-coordinate of the data point), one layer with 500 nodes, and one output layer. "data" is what I feed to the neural network. 


def neural_network_model(data):

	#The weights are randomized for the first run through. The weights will be calibrated to minimize the cost function. 
	hidden_1_layer={'weights':tf.Variable(tf.random_normal([2,num_Nodes])),'biases':tf.Variable(tf.random_normal([num_Nodes]))}
	
	output_layer={'weights':tf.Variable(tf.random_normal([num_Nodes,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}



	
	#This preforms the actual manipulation of the data. The data is multiplied by the weights, and then the biases are added. This gives the nodes something to work with (for their activation function)
	l1=tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
	
	#Takes the output of the (input*weights) + biases of layer one and puts it through a sigmoid function (to get the input to the output nodes). This is the output of the first layer that is to be fed into the output layer. The explicit function is: 1/(1+exp(-x))
	l1=tf.nn.sigmoid(l1)


	#Gives the two final values of the ANN. The first output value corresponds to the input point being part of the circle set and the second output value corresponds to the input point being part of the torus set. The two values are not between 0 and 1. The larger value indicates that the input is more likely to belong to one set as opposed to the other. 
	output=tf.add(tf.matmul(l1,output_layer['weights']),output_layer['biases'])
	
	return output




#STEP THREE: TRAINING THE NEURAL NETWORK/USING THE NEURAL NETWORK


#x is the data that is to be fed through. 	
def train_neural_network(x):


	#THIS PART IS SETTING UP OPERATIONS ON THE NEURAL NETWORK
	#Getting the output of the ANN
	prediction=neural_network_model(x)
	
	#Setting up the cost function (which is later minimized to increase accuracy).
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
	
	#Optimizing function that acts to minimize the cost function (and therefore minimize the difference between the output values and the expected values)
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	#One epoch is a complete cycle through the data as well as the backpropogation (adjusting the weights to increase accuracy)
	num_epochs=10

	#Now actually running the ANN.
	#Running the ANN that was established earlier.
	with tf.Session() as sess:

		#Initializes the variables (the weights and biases of the first hidden layer and the output layer) that were set up earlier. 
		sess.run(tf.global_variables_initializer())
		
		#THIS IS THE TRAINING
		for epoch in range(num_epochs):
			epoch_loss = 0
			i=0
			batch_num=0
			
			#Takes batches of the training data.
			while i < len(train_x):
				start=i
				end=i+batch_size
				batch_x=np.array(train_x[start:end])
				batch_y=np.array(train_y[start:end])



				#This is where the ANN is actually ran. The weights and biases are modified with this line of code (since the optimizer operation is ran).
				_,c,ToPrint=sess.run([optimizer,cost, prediction], feed_dict={x: batch_x, y: batch_y})
				epoch_loss+=c
				correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
                		print "Accuracy within batch: %f" %(accuracy.eval({x:test_x, y:test_y}))
				i+=batch_size
			epochNum=epoch+1

			#Getting the accuracy per epoch
			correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
                	accuracy=tf.reduce_mean(tf.cast(correct,'float'))
                	print "Accuracy: %f" %(accuracy.eval({x:test_x, y:test_y}))

			#Outputs a message to indicate the cost function and accuracy. This enables you to make sure the cost function is decreasing and the accuracy is increasing.
			print "Epoch %d completed out of %d. Loss: %f" %(epoch+1,num_epochs,epoch_loss)



		#THIS IS THE VALIDATION. Since the optimizer operation is not being ran, the weights are not being changed. 


		#Saving the output of the seperate data sets
		#Circle_2000_Output=sess.run([prediction],feed_dict={x:Circle_2000})
		#with open ("/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ANNOutput/RawOutput/SignificantOverlap/CircleRawOutput/Circle_2000_Output.txt", 'w') as C2000:
			#Circle_2000_Output=np.asarray(Circle_2000_Output)
			#Circle_2000_Output=np.reshape(Circle_2000_Output,(2000,2))
			#np.savetxt(C2000, Circle_2000_Output, fmt='%1.8f', delimiter=",")





		#Circle_8000_Output=sess.run([prediction],feed_dict={x:Circle_8000})
		#with open ("/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ANNOutput/RawOutput/SignificantOverlap/CircleRawOutput/Circle_8000_Output.txt", 'w') as C8000:
			#Circle_8000_Output=np.asarray(Circle_8000_Output)
			#Circle_8000_Output=np.reshape(Circle_8000_Output,(8000,2))
			#np.savetxt(C8000, Circle_8000_Output, fmt='%1.8f', delimiter=",")

	
	

		#Torus_2000_Output=sess.run([prediction],feed_dict={x:Torus_2000})
		#with open ("/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ANNOutput/RawOutput/SignificantOverlap/TorusRawOutput/Torus_2000_Output.txt", 'w') as T2000:
			#Torus_2000_Output=np.asarray(Torus_2000_Output)
			#Torus_2000_Output=np.reshape(Torus_2000_Output,(2000,2))
			#np.savetxt(T2000, Torus_2000_Output, fmt='%1.8f', delimiter=",")



		#Torus_8000_Output=sess.run([prediction],feed_dict={x:Torus_8000})	
		#with open ("/Users/Will/Research/tensorflow/MakingHistogramsV3/Data/ANNOutput/RawOutput/SignificantOverlap/TorusRawOutput/Torus_8000_Output.txt", 'w') as T8000:
			#Torus_8000_Output=np.asarray(Torus_8000_Output)
			#Torus_8000_Output=np.reshape(Torus_8000_Output,(8000,2))
			#np.savetxt(T8000, Torus_8000_Output, fmt='%1.8f', delimiter=",")


		#This gives the final accuracy of the ANN (as computed based on the accuracy when the testing data is passed through).
		correct=tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print "Accuracy: %f" %(accuracy.eval({x:test_x, y:test_y}))


#Running the training/validation code.
train_neural_network(x) 
