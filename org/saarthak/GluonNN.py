import mxnet as mx
from mxnet import gluon, autograd, ndarray
import numpy as np

# Training the datasets with "mx.gluon.data.DataLoader".
train_dataset = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                      batch_size=32, shuffle=True)
test_dataset = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=lambda data, label: (data.astype(np.float32)/255, label)),
                                     batch_size=32, shuffle=False)
net = gluon.nn.Sequential()

# Model architecture with 256 and 128 nodes.
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu"))  # first layer with 256 nodes
    net.add(gluon.nn.Dense(128, activation="relu"))  # second layer with 256 nodes
    net.add(gluon.nn.Dense(10))


# if you wish to use the HybridSequential you can use the following commented blocks.
# comment out the sequential block comments before doing so.
#####################################HybridSequential###################################################
# net = gluon.nn.HybridSequential
# with net.name_scope():
#    net.add(nn.Dense(256, activation="relu"))
#   net.add(nn.Dense(128, activation="relu"))
#    net.add(nn.Dense(2))
#####################################HybridSequential###################################################

    net.collect_params().initialize(mx.init.Normal(sigma=0.05))  # SD of 0.05 for Normal distribution

    # softmax cross entropy loss function to measure the accuracy of the model
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    # stochastic gradient descent training algorithm.
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

    # Executing the training loop.
    epochs = 10
    for e in range(epochs):
        for i, (data, label) in enumerate(train_dataset):
            # You can choose between your cpu amd gpu for your computation.
            data = data.as_in_context(mx.cpu()).reshape((-1, 784))
            label = label.as_in_context(mx.cpu())
            with autograd.record():  # Start recording the derivatives
                output = net(data)  # the forward iteration
                loss = softmax_cross_entropy(output, label)
                loss.backward()
            trainer.step(data.shape[0])

            curr_loss = ndarray.mean(loss).asscalar()

        # Provide stats on the improvement of the model over each epoch.
        # with each epoch, you will notice that accuracy improves.
        print("Epoch {}. Current Loss: {}.".format(e, curr_loss))


