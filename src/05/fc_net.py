from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class ThreeLayerNet(object):
    """
    A three-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with              #
        # first -> second layer weights and biases using the keys 'W1' and 'b1'    #
        # and second -> third layer weights and biases using the keys 'W2' and 'b2'.#                                 #
        ############################################################################
        std = weight_scale
        self.params['W1'] = np.random.normal(scale=std, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(scale=std, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)
             
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        
        
        ############################################################################
        # TODO: Implement the forward pass for the three-layer net, computing the  #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        arf_out, arf_cache = affine_relu_forward(X, W1, b1)
        scores, af_cache = affine_forward(arf_out,W2,b2)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer net. Store the loss#
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        ############################################################################
        reg = self.reg
        num_train = X.shape[0]
        loss, dx1 = softmax_loss(scores, y)
        loss += 0.5*reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        
        ab_dx, ab_dw, ab_db = affine_backward(dx1, af_cache)
        grads['W2'] = ab_dw + 2* reg * W2
        grads['b2'] = ab_db
    
        arb_dx, arb_dw, arb_db = affine_relu_backward(ab_dx, arf_cache)
        grads['W1'] = arb_dw + 2* reg * W1
        grads['b1'] = arb_db


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. For a network with L layers,
    the architecture will be

    {affine - relu} x (L - 2) - affine - softmax

    where the {...} block is repeated L - 2 times, i.e., number of hidden layers

    Similar to the ThreeLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first ->    #
        # secnd layer in W1 and b1; for the second -> third layer use W2 and b2,   #
        # etc. Weights should be  initialized from a normal distribution with      #
        # standard deviation equal to weight_scale and biases should be            #
        # initialized to zero.                                                     #
        ############################################################################

        num_w = len(hidden_dims)
        std = weight_scale
        for i in range(num_w):
            if i==0:
                self.params['W'+str(i+1)] = np.random.normal(scale=std, size=(input_dim, hidden_dims[i]))
                self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
            else:
                self.params['W'+str(i+1)] = np.random.normal(scale=std, size=(hidden_dims[i-1], hidden_dims[i]))
                self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
                    

        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as ThreeLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'


        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        ############################################################################
        arf_cache={}
        num_w = self.num_layers-1
        scores = X
        for i in range(num_w):
            W, b = self.params['W'+str(i+1)], self.params['b'+str(i+1)]
            if i == num_w-1:
                scores, af_cache = affine_forward(scores,W,b)
            else:
                scores, arf_cache[i] = affine_relu_forward(scores,W,b)
                

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        reg = self.reg
        loss, dx1 = softmax_loss(scores, y)
        s=0
        for i in range(num_w):
            s += np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
        loss += reg * s
        
        for i in reversed(range(num_w)):
            if i==num_w-1:
                ab_dx, ab_dw, ab_db = affine_backward(dx1, af_cache)
                grads['W'+str(i+1)] = ab_dw + 2* reg * self.params['W'+str(i+1)]
                grads['b'+str(i+1)] = ab_db
            else:
                arb_dx, arb_dw, arb_db = affine_relu_backward(ab_dx, arf_cache[i])
                grads['W'+str(i+1)] = arb_dw + 2* reg * self.params['W'+str(i+1)]
                grads['b'+str(i+1)] = arb_db


        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
