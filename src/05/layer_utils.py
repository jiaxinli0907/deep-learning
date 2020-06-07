from layers import *

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """  
    
    ###########################################################################
    # TODO: Implement the affine_relu forward pass                            #
    ###########################################################################

    af_out, af_cache = affine_forward(x,w,b)
    out, rf_cache = relu_forward(af_out)
    cache = (af_cache, rf_cache)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    ###########################################################################
    # TODO: Implement the affine_relu backward pass                            #
    ###########################################################################
    af_cache, rf_cache = cache
    rb_dx = relu_backward(dout, rf_cache)
    dx,dw,db = affine_backward(rb_dx, af_cache)

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx, dw, db
