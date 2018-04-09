import numpy as np

def sigmoid(x, derivative=False):
	if derivative:
		return x*(1-x)
	else:
		return (1/(1+np.exp(-x)))

def init_weights(out_size, in_size):
	W = np.random.rand(out_size, in_size)*2 - 1
	return W

def forward_pass(W, X, b):
	H = W.dot(X.T)+b
	return H

X = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
Y = np.array([0, 1, 1, 0])

W0 = init_weights(2, 2)
W1 = init_weights(2, 2)
W2 = init_weights(1, 2)
'''
print (W0)
print (W1)
print (W2)
'''
bias0 = np.zeros((2,1))
bias1 = np.zeros((2,1))
bias2 = np.zeros((1,1))

epochs = 100001
learning_rate = 0.1

for i in range (epochs):
	total_loss = 0
	for j in range(4):
		# forward pass to get the output
		L1 = sigmoid(forward_pass(W0,X[j],bias0))
		L2 = sigmoid(forward_pass(W1,L1.T,bias1))	#both L1, L2 are shape(2,1)
		Y_hat = sigmoid(forward_pass(W2,L2.T,bias2))
		loss = abs(Y_hat - Y[j])
		total_loss = total_loss + loss

		#backpropagation
		delta3 = (Y_hat - Y[j])*sigmoid(Y_hat,derivative=True)
		delta2 = delta3*np.multiply(W2.T,sigmoid(L2,derivative=True))
		delta1 = np.multiply(np.dot(W1,delta2),sigmoid(L1,derivative=True))

		#update weights
		W2 = W2 - learning_rate*delta3*L2.T
		W1 = W1 - learning_rate*np.dot(delta2,L1.T)
		W0 = W0 - learning_rate*np.dot(delta1,X[j])

		bias2 = bias2 - learning_rate*delta3
		bias1 = bias1 - learning_rate*delta2
		bias0 = bias0 - learning_rate*delta1
	total_loss = total_loss/4
	if i % 10000 == 0:
		print ("epochs: ",i,"     loss is: ",total_loss)
	'''
	if total_loss < 0.09:
		break
	'''
# print predict values
for j in range(4):
	# forward pass to get the output
	L1 = sigmoid(forward_pass(W0,X[j],bias0))
	L2 = sigmoid(forward_pass(W1,L1.T,bias1))	#both L1, L2 are shape(2,1)
	Y_hat = sigmoid(forward_pass(W2,L2.T,bias2))
	print ("the answer of : ",X[j],"is :" ,Y_hat)