from __future__ import print_function
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from mxnet import gluon
import matplotlib.pyplot as plt



# context to tell gluon where to do most of the computation. Chooose between CPU and GPU.
cntx = mx.cpu()


# Building random dataset.
num_inputs = 2
num_outputs = 1
num_examples = 10000


def real_fn(X):
    return 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
noise = 0.01 * nd.random_normal(shape=(num_examples,))
y = real_fn(X) + noise


# Using Gluon to DataLoader to handle our data batching.
batch_size = 4
train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(X, y),
                                      batch_size=batch_size, shuffle=True)


net = gluon.nn.Sequential()

with net.name_scope():
    net.add(gluon.nn.Dense(1, in_units=2))

    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=cntx)

# Using Gluon loss function.
square_loss = gluon.loss.L2Loss()

# stochastic gradient descent training algorithm.
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})


# Executing the training loop.
# Generating predictions (yhat) and the loss (loss) by executing a forward pass through the network.
# Calculating gradients by making a backwards pass through the network via loss.backward()
epochs = 1
smoothing_constant = .01
moving_loss = 0
niter = 0
loss_seq = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(cntx)
        label = label.as_in_context(cntx)
        with autograd.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        # Keeping the losses in motion.
        niter +=1
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (1 - smoothing_constant) * moving_loss + (smoothing_constant) * curr_loss

        # correcting the bias from the moving averages
        est_loss = moving_loss/(1-(1-smoothing_constant)**niter)
        loss_seq.append(est_loss)

    print("Epoch %s. Moving avg of MSE: %s" % (e, est_loss))

# Obtaining the learned model parameters
params = net.collect_params()  # this returns a ParameterDict
print('The type of "params" is a ',type(params))

for param in params.values():
    print(param.name,param.data())














