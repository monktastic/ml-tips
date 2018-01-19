"""
Demonstrates how to call a remote fn (or really, invoke arbitrary python code)
and then use that layer's relative gradient (d_output / d_input) to generate
the complete gradient d_Cost / d_input.

This code uses that "remote" fn in a layer between two trainable layers, just
to demonstrate that backprop works.
"""

from functools import partial

import numpy as np
import tensorflow as tf
from keras import constraints
from keras import losses
from keras.layers.core import Lambda, Dense
from keras.models import Sequential
from keras.optimizers import Adam


# Gross, but we need a sess for call_fn to evaluate the gradient. When calling
# the actual Fn, we won't need this.
sess = tf.Session()


# With a little help from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342#gistcomment-2070556

# Define custom py_func which takes also a grad op as argument:
def get_py_func(fn, grad_fn):
    def py_func(x):
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad_fn)
        g = tf.get_default_graph()
        # See https://stackoverflow.com/questions/41391718/tensorflows-gradient-override-map-function
        # The c++ op is called PyFunc. Here we're overriding its gradient function
        # with the one we registered above, under the name rnd_name. We also
        # override PyFuncStateless in case stateful=False.
        with g.gradient_override_map({"PyFunc": rnd_name,
                                      "PyFuncStateless": rnd_name}):
            y, grad = tf.py_func(fn, [x], [tf.float32, tf.float32], stateful=False)
            y.set_shape(x.get_shape())
            return y
    return py_func


# Simulates a call to fn with given model and its gradients.
# Actually just does the work locally.
def call_fn(x, model, grads):
    y = model.predict(x)
    grad_val = grads[0].eval(session=sess, feed_dict={model.input: x})
    return y, grad_val


# If you want to run arbitrary python code on backprop, use this in a tf.py_func
# in calc_grad. Currently unused.
def grad_pyfunc(dC_dout, dout_din):
    # Could call remote fn here to pass along gradient info.
    return dC_dout * dout_din


# Custom gradient function. Requires that the forward pass stashed its dout/din
# in outputs[1].
def calc_grad(op, dC_dout0, dC_dout1):
    return tf.py_func(grad_pyfunc, [dC_dout0, op.outputs[1]], tf.float32)

    # Since outputs[1] is dout0/din, this returns dC/dout0 * dout0/din =
    # dC/din as required.
    # return dC_dout0 * op.outputs[1]


# A simple function that our fn model will calculate. Factored out so that
# our test data generator can call it too.
def fn_layer_func(x):
    return 2 * x + [50, 100]


# Generator to use with fit_generator.
def gen():
    while True:
        x = np.random.random_integers(0, 100, size=2).reshape((2,)).astype(np.float32)
        y = fn_layer_func(x)
        # Add batch dimension.
        yield ([np.expand_dims(x, 0)], [np.expand_dims(y, 0)])


# Simple model for our simulated fn call to use.
def get_fn_model():
    fn_model = Sequential()
    fn_model.add(Lambda(fn_layer_func, name='fn-lambda', input_shape=(2,)))
    return fn_model


def get_local_model(remote_func):
    # This is our local model. It's basically:
    #
    #   Dense with no bias -> fn -> dense with no bias
    #
    # The idea is for the dense layers to learn the identity function,
    # since fn does all the work (notice how the generator computes the
    # same function as fn_model).
    model = Sequential()
    model.add(Dense(
        2,
        bias_constraint=constraints.MaxNorm(0),
        kernel_constraint=constraints.NonNeg(),
        input_shape=(2,)))

    model.add(Lambda(remote_func))

    model.add(Dense(
        2,
        bias_constraint=constraints.MaxNorm(0),
        kernel_constraint=constraints.NonNeg()))

    return model


if __name__ == "__main__":
    # Get the model to be called by our py_func.
    fn_model = get_fn_model()

    # Somehow, calling predict() here prevents this error when call_fn tries
    # to call predict():
    #  Tensor Tensor("fn-lambda/add:0", shape=(?, 2), dtype=float32) is not an element of this graph.
    x = np.array([[20, 100]], dtype=np.float32)
    pred = fn_model.predict(x)

    # Get a version of call_fn that has our model and its gradients. Generating
    # the grads tensor in the python function takes forever.
    grads = tf.gradients(fn_model.outputs, fn_model.inputs)
    call_fn_on_model = partial(call_fn, model=fn_model, grads=grads)
    # Construct the py_func.
    py_func = get_py_func(call_fn_on_model, calc_grad)

    # Create the local model.
    model = get_local_model(py_func)

    # Train the local model, ensuring backprop through fn works.
    model.compile(
        optimizer=Adam(lr=0.1, decay=0.03),
        loss=losses.mean_absolute_error
    )
    model.fit_generator(
        gen(),
        steps_per_epoch=1000,
        epochs=12
    )

    # The learned kernels should multiply out to Identity.
    k1 = model.layers[0].get_weights()[0]
    k2 = model.layers[2].get_weights()[0]
    k_prod = np.dot(k1, k2)

    print(k_prod)

    dist = np.linalg.norm(k_prod - np.eye(2, 2))
    print("Distance to Identity matrix:", dist)

