from functools import partial

import numpy as np
import tensorflow as tf
from keras import constraints
from keras import losses
from keras.layers.core import Lambda, Dense
from keras.models import Sequential
from keras.optimizers import Adam


# With a little help from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342#gistcomment-2070556

# Define custom py_func which takes also a grad op as argument:
def get_py_func(fn, grad_fn):
    def py_func(x):
        # Need to generate a unique name to avoid duplicates:
        rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

        tf.RegisterGradient(rnd_name)(grad_fn)
        g = tf.get_default_graph()
        with g.gradient_override_map({"PyFunc": rnd_name,
                                      "PyFuncStateless": rnd_name}):
            y = tf.py_func(fn, [x], tf.float32)
            y.set_shape(x.get_shape())
            return y
    return py_func


# InvalidArgumentError (see above for traceback): 0-th value returned by pyfunc_0 is double, but expects float
#	 [[Node: lambda_1/PyFuncStateless = PyFuncStateless[Tin=[DT_FLOAT], Tout=[DT_FLOAT], _gradient_op_type="PyFuncGrad61541397", token="pyfunc_0", _device="/job:localhost/replica:0/task:0/cpu:0"](dense_1/BiasAdd)]]
def fwd_func(x):
	ret = 2 * x + [50, 100]
	return ret.astype(np.float32)


def grad_func(op, dC_dout0):
	return dC_dout0 * 2


# Generator to use with fit_generator.
def gen():
    while True:
        x = np.random.random_integers(0, 100, size=2).reshape((2,)).astype(np.float32)
        y = fwd_func(x)
        # Add batch dimension.
        yield ([np.expand_dims(x, 0)], [np.expand_dims(y, 0)])



if __name__ == "__main__":
    model = Sequential()
    model.add(Dense(
        2,
        bias_constraint=constraints.MaxNorm(0),
        kernel_constraint=constraints.NonNeg(),
        input_shape=(2,)))

    py_func = get_py_func(fwd_func, grad_func)
    model.add(Lambda(py_func))
    return model


    # Train the local model, ensuring backprop works.
    model.compile(
        optimizer=Adam(lr=0.1, decay=0.03),
        loss=losses.mean_absolute_error
    )
    model.fit_generator(
        gen(),
        steps_per_epoch=1000,
        epochs=12
    )

    weights = model.layers[0].get_weights()[0]
    dist = np.linalg.norm(weights - np.eye(2, 2))
    print("Distance to Identity matrix:", dist)

