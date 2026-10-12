import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from data import X_train, y_train, X_val, y_val

class Layer_dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * cp.random.randn(n_inputs, n_neurons)
        self.biases = cp.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = cp.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = cp.dot(self.inputs.T, dvalues)
        self.dbiases = cp.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = cp.dot(dvalues, self.weights.T)


class Activation_ReLu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = cp.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        self.inputs = inputs
        exp_values = cp.exp(inputs - cp.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/ cp.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        data_loss = cp.mean(sample_loss)
        return data_loss
    def forward(self, output, y):
        probabilities = output[cp.arange(len(y)), y]
        neg_log = -cp.log(probabilities)
        return neg_log

class Softmax_Loss:
    def backward(self, dvalues, y):
        self.dinputs = dvalues.copy()
        self.dinputs[cp.arange(len(y)), y] -=1
        self.dinputs = self.dinputs/ len(y)

class Optimizer_SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum

    def update_params(self, layer):

        if not hasattr(layer, "weight_momentums"):
            layer.weight_momentums = cp.zeros_like(layer.weights)
            layer.bias_momentums = cp.zeros_like(layer.biases)

        layer.weight_momentums = (
            self.momentum * layer.weight_momentums
            - self.learning_rate * layer.dweights
        )

        layer.bias_momentums = (
            self.momentum * layer.bias_momentums
            - self.learning_rate * layer.dbiases
        )

        layer.weights += layer.weight_momentums
        layer.biases += layer.bias_momentums

def create_batches(X, y, batch_size=128):
    X = cp.asarray(X)
    y = cp.asarray(y)
    indices = cp.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    batches = []

    for start in range(0, len(X), batch_size):
        X_batch = X[start:start + batch_size]
        y_batch = y[start:start + batch_size]
        batches.append((X_batch, y_batch))
    return batches

def calculate_accuracy(X, y):
    X = cp.asarray(X)
    y = cp.asarray(y)
    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    activation2.forward(layer2.output)

    layer3.forward(activation2.output)
    activation3.forward(layer3.output)

    predictions = cp.argmax(activation3.output, axis=1)
    accuracy = cp.mean(predictions == y)

    return float(accuracy)

batches = create_batches(X_train, y_train, batch_size=128)

inputs, y_true = batches[0]

activation1 = Activation_ReLu()
activation2 = Activation_ReLu()
activation3 = Activation_Softmax()

layer1 = Layer_dense(784, 286)
layer2 = Layer_dense(286, 286)
layer3 = Layer_dense(286, 10)

loss_function = Loss()
softmax_Loss = Softmax_Loss()

optimizer = Optimizer_SGD(learning_rate=0.0035, momentum=0.92)

if __name__ == "__main__":
    loss_history = []
    validation_accuracies = []
    for epochs in range(10):
        for batch_index, (inputs, y_true) in enumerate(batches):
            layer1.forward(inputs)
            activation1.forward(layer1.output)

            layer2.forward(activation1.output)
            activation2.forward(layer2.output)

            layer3.forward(activation2.output)
            activation3.forward(layer3.output)

            loss = loss_function.calculate(activation3.output, y_true)
            loss_history.append(float(loss))

            softmax_Loss.backward(activation3.output, y_true)
            layer3.backward(softmax_Loss.dinputs)

            activation2.backward(layer3.dinputs)
            layer2.backward(activation2.dinputs)

            activation1.backward(layer2.dinputs)
            layer1.backward(activation1.dinputs)

            optimizer.update_params(layer1)
            optimizer.update_params(layer2)
            optimizer.update_params(layer3)
        accuracy = calculate_accuracy(X_val, y_val)
        validation_accuracies.append(accuracy * 100)
        print("Generation: ", epochs)
        print("Pertes ", loss)
        print(f"Accuracy : {accuracy * 100:.2f}%")

    cp.savez(
    "model1-4.npz",
    w1=layer1.weights,
    b1=layer1.biases,
    w2=layer2.weights,
    b2=layer2.biases,
    w3=layer3.weights,
    b3=layer3.biases
    )

    print("Poids du réseau sauvegardés !")


    fig, ax1 = plt.subplots(figsize=(20, 12))

    # Courbe de loss : axe de gauche
    ax1.plot(loss_history, color="royalblue", alpha=0.7, label="Pertes entraînement")
    ax1.set_xlabel("Numéro du groupe")
    ax1.set_ylabel("Pertes", color="royalblue")
    ax1.tick_params(axis="y", labelcolor="royalblue")
    ax1.grid(True, alpha=0.3)

    # Courbe d'accuracy : axe de droite
    ax2 = ax1.twinx()

    epochs_x = range(1, len(validation_accuracies) + 1)

    ax2.plot(
        [((e - 1) * len(batches)) for e in epochs_x],
        validation_accuracies,
        color="darkorange",
        marker="o",
        linewidth=2,
        label="Accuracy"
    )

    ax2.set_ylabel("Accuracy (%)", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    # Titre et légende
    fig.suptitle("Apprentissage du réseau MNIST")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    plt.show()

