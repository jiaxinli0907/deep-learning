from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input layer - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the third fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: Weights from input layer to fully-connected layer; has shape (D, H)
        b1: Biases from input layer to fully-connected layer; has shape (H,)
        W2: Weights from fully-connected layer to output layer; has shape (H, C)
        b2: Biases from fully-connected layer to output layer; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a three layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # Compute the forward pass
        scores = None
        #############################################################################
        # TASK 1
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of     #
        # shape (N, C).                                         #
        #############################################################################
        z2 = X.dot(W1) + b1
        a2 = np.maximum(z2,0)
        scores = a2.dot(W2) + b2
 
        #############################################################################
        #                   END OF YOUR CODE                     #
        #############################################################################
    
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        
        
        #############################################################################
        # TASK 2
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax        #
        # classifier loss.                                       #
        #############################################################################
   
        scores -= np.max(scores,axis=1,keepdims=True)
        p = np.exp(scores)/np.sum(np.exp(scores),axis=1,keepdims=True)
        loss = -np.sum(np.log(p[range(N),y]))
        loss /= N
        loss += reg * (np.sum(W1 * W1)+np.sum(W2 * W2))
  
        #############################################################################
        #                   END OF YOUR CODE                      #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TASK 3
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,     #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        
        p[range(N),y] -= 1
  

        delta3 = p/N 
        grads['W2'] = a2.T.dot(delta3) + reg*W2
        grads['b2'] = np.sum(delta3,axis=0)
        
        
        delta2 = delta3.dot(W2.T)
        
        relu2 = np.maximum(a2, 0)
        relu2[relu2>0] = 1
        delta2 *= relu2
        
        grads['W1'] = X.T.dot(delta2) + reg*W1
        grads['b1'] = np.sum(delta2,axis=0)
      

        #############################################################################
        #              END OF YOUR CODE                          #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TASK 4a
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                    #
            #########################################################################
            
            p = np.random.choice(num_train, batch_size)
            X_batch = X[p,:]
            y_batch = y[p]
         
        
            #########################################################################
            #                     END OF YOUR CODE                 #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TASK 4b
            # TODO: Use the gradients in the grads dictionary to update the       #
            # parameters of the network (stored in the dictionary self.params)     #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                  #
            #########################################################################
            self.params['W1'] -= learning_rate*grads['W1']
            self.params['W2'] -= learning_rate*grads['W2']
            self.params['b1'] -= learning_rate*grads['b1']
            self.params['b2'] -= learning_rate*grads['b2']
            #########################################################################
            #                 END OF YOUR CODE                     #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
           }

    def predict(self, X):
        """
        Use the trained weights of this three-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TASK 4c
        # TODO: Implement this function; it should be VERY simple!           #
        ###########################################################################
        a2 = X.dot(self.params['W1']) + self.params['b1']
        z2 = np.maximum(a2,0)
        a3 = z2.dot(self.params['W2']) + self.params['b2']
        y_pred = np.argmax(a3,axis=1)

        ###########################################################################
        #                    END OF YOUR CODE                   #
        ###########################################################################

        return y_pred
