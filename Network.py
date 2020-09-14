from Tensor import Tensor
from Tensor import product
import random
import pickle
import os

class InputLayer:

    layer_type = "input"

    def __init__(self, tensor, reshape=False, activation=Tensor.dummy):
        self.reshape = reshape
        self.activation = activation
        self.forward_pass(tensor)

    def forward_pass(self, tensor):
        if self.reshape:
            tensor = tensor.reshape(tensor.dims[:1]+self.reshape)
        self.tensor = tensor
        self.result_dims = tensor.dims
        return tensor


class ConvolutionalLayer:

    layer_type = "convolutional"

    def __init__(self, kernel_size, kernel_count, prev_layer, stride=1,
                 activation=Tensor.leaky_relu):
        self.dims = [kernel_count, kernel_size, kernel_size]
        self.tensor = Tensor(dims=self.dims)
        self.activation = activation
        self.result_dims = Tensor.convolve_dims(prev_layer.result_dims,
                                                self.tensor.dims, stride)
        
    def forward_pass(self, tensor_in):
        tensor_out, mask = tensor_in.convolve(self.tensor)
        tensor_out = self.activation(tensor_out)
        self.result_dims = tensor_out.dims
        self.mask = mask
        return tensor_out

    def backward_pass(self, delta_in, hidden, learning_rate=1):
        delta_in = delta_in.reshape(self.result_dims)
        t = hidden.reverse_convolve(delta_in, fs=self.tensor.dims[-1])
        self.tensor += t*learning_rate


class FullyConnectedLayer:

    layer_type = "fully_connected"

    def __init__(self, width, prev_layer, activation=Tensor.tanh,
                 array=False, drop_out=0):
        self.dims = [product(prev_layer.result_dims[1:]), width]
        self.drop_out_percentage = drop_out
        if array:
            self.tensor = Tensor(array=array)
        else:
            self.tensor = Tensor(dims=self.dims)
        self.result_dims = [prev_layer.result_dims[0], width]
        self.prev_layer = prev_layer
        self.activation = activation

    def forward_pass(self, tensor_in, train=True):
        if len(tensor_in.dims)<3:
            resh = tensor_in
        else:
            new_dims = [tensor_in.dims[0], product(tensor_in.dims[1:])]
            resh = tensor_in.reshape(new_dims)
        if train:
            dims = [tensor_in.dims[0], self.dims[-1]]
            mask = Tensor.generate_zero_mask(dims, self.drop_out_percentage)
            return self.activation(resh.dot(self.tensor))*mask
        else:
            return self.activation(resh.dot(self.tensor))

    def backward_pass(self, delta_in, hidden, learning_rate,
                      momentum=0, drop=True):
        hidden = hidden.reshape([hidden.dims[0], product(hidden.dims[1:])])
        transpose = self.tensor.transpose()
        p = (100-self.drop_out_percentage)/100
        delta = delta_in.dot(transpose)*self.activation(hidden, deriv=True)
        update = (hidden.transpose().dot(delta_in))*learning_rate*p
        momentum_update = update+momentum
        self.tensor += momentum_update
        self.tensor = self.tensor.max_norm_constrain()
        return delta, update
 

class OutputLayer(FullyConnectedLayer):

    layer_type = "output_layer"

    def __init__(self, outputs, prev_layer,
                 activation=Tensor.tanh, array=False):
        super().__init__(outputs, prev_layer, activation, array)

    def forward_pass(self, tensor_in, train=False):
        return super().forward_pass(tensor_in, train=False).soft_max()

    def backward_pass(self, delta_in, hidden, learning_rate, momentum=0):
        return super().backward_pass(delta_in, hidden, learning_rate,
                                     momentum, False)


class MaxPoolLayer:

    layer_type = "max_pool"

    def __init__(self, prev_layer, window_size=2, stride=2):
        self.result_dims = prev_layer.result_dims[:-2]
        self.result_dims.append(prev_layer.result_dims[-2]//stride)
        self.result_dims.append(prev_layer.result_dims[-1]//stride)
        self.window_size = window_size
        self.stride = stride

    def forward_pass(self, tensor_in):
        result, self.mask = tensor_in.max_pool(self.window_size, self.stride)
        self.result_dims = result.dims
        return result

    def backward_pass(self, delta_in, hidden, learning_rate=1):
        delta_in = delta_in.reshape(self.result_dims)
        delta = delta_in.reverse_max_pool(self.mask, self.window_size, self.stride)
        return delta


class Network:

    def __init__(self, momentum_decay=0.9):
        self.layers = []
        self.high_score = 0
        self.test_cycles = 0
        self.learning_rate = 1
        self.momentum_decay = momentum_decay

    def finalise(self):
        self.updates = []
        for layer in self.layers[1:][::-1]:
            self.updates.append(Tensor.generate_zero_mask(layer.tensor.dims,
                                                          100))

    def add_layer(self, layer):
        self.layers.append(layer)

    def convolutional_layers(self):
        result = []
        for i, l in enumerate(self.layers):
            if l.layer_type=="convolutional":
                result.append((i, l.tensor))
        return result

    def max_layers(self):
        result = []
        for i, l in enumerate(self.layers):
            if l.layer_type=="max_pool":
                result.append(i)
        return result

    def forward_pass(self, tensor=False, train=True):
        pass_results = [self.layers[0].forward_pass(tensor)]
        for i, layer in enumerate(self.layers[1:]):
            pass_results.append(layer.forward_pass(pass_results[i],
                                                   train=train))
        self.pass_results = pass_results

    def backward_pass(self, tensor, batch):
        error = tensor-self.pass_results[-1]
        delta = error*self.pass_results[-1]
        zipped = zip(self.layers[::-1], self.pass_results[:-1][::-1])
        for i, (layer, hidden) in enumerate(zipped):
            delta, update = layer.backward_pass(delta,
                                                hidden, self.learning_rate/batch,
                                                self.updates[i])
            self.updates[i] = self.updates[i]*self.momentum_decay+update*0.6

    def batch(self, inputs, outputs, batch_size, same=False):
        inputs = inputs.array if isinstance(inputs, Tensor) else inputs
        outputs = inputs.array if isinstance(inputs, Tensor) else outputs
        in_array, out_array = [], []
        for i in range(batch_size):
            r = i if same else random.randrange(len(inputs))
            in_array.append(inputs[r])
            out_array.append(outputs[r])
        return (Tensor(array=in_array), Tensor(array=out_array))
            
    def train(self, inputs, outputs, learning_rate=0.1, cycles=25, batch=25):
        self.learning_rate = learning_rate
        for i in range(cycles):
            print(i, end=" ")
            tensor_in, tensor_out = self.batch(inputs, outputs, batch)
            self.forward_pass(tensor_in)
            self.backward_pass(tensor_out, batch)
        self.test_cycles += 1      

    def test(self, inputs, outputs, batch=100):
        tensor_in, tensor_out = self.batch(inputs, outputs, batch, True)
        predicted = self.predict(tensor_in)
        score = 0
        for a, b in zip(predicted.array, tensor_out.array):
            score += 1 if a==b else 0
        return score/batch*100

    def predict(self, tensor):
        self.forward_pass(tensor, train=False)
        return self.pass_results[-1].one_hot()

    def save(self, score=0):
        if score>=self.high_score:
            try:
                os.remove("Model{0}.Pickle".format(self.high_score))
            except:
                pass
            pickle.dump((self.layers, score),
                        open("Model{0}.Pickle".format(score), "wb"))
            self.high_score = score

    def load(self, score=0):
        self.layers, self.high_score = pickle.load(open("Model{0}.pickle"
                                                        .format(score), "rb"))


filters = [
[[-1 ,-1 ,-1 ,-1 ,-1 ],
 [0.6,0.6,0.6,0.6,0.6],
 [1  ,1,  1,  1,  1, ],
 [0.6,0.6,0.6,0.6,0.6],
 [-1 ,-1 ,-1 ,-1 ,-1]
    ]]

if __name__ == "__main__":
    run = True
else:
    run = input("Run network code") in ["YES", "y", "Y", "yes", "Yes"]

if run:
    network = Network(0.6)
    path = "/Users/user/Desktop/Tensor/"
    images = pickle.load(open("{0}test_images.pickle".format(path), "rb"))
    labels = pickle.load(open("{0}test_labels.pickle".format(path), "rb"))
    image_tensor = Tensor(array=images[:25])
    l0 = InputLayer(image_tensor)
    l1 = FullyConnectedLayer(120, l0, drop_out=20)
    l2 = FullyConnectedLayer(80, l1, drop_out=20)
    l3 = OutputLayer(10, l2)
    for l in [l0, l1, l2, l3]:
        network.add_layer(l)
    network.finalise()
    score = 0
    while True:
        print("TESTING...", end="")
        score = network.test(images, labels, 500)
        print("Score: {0}%".format(score))
        network.train(images, labels, (1-(100-score)/100)/80, 4, 100)
        print()
        network.save(score)





















