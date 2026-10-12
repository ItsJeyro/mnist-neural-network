from training import Layer_dense, Activation_ReLu, Activation_Softmax, Loss, Optimizer_SGD, create_batches
from data import X_test, y_test
import cupy as cp
layer1 = Layer_dense(784, 286)
layer2 = Layer_dense(286, 286)
layer3 = Layer_dense(286, 10)

activation1 = Activation_ReLu()
activation2 = Activation_ReLu()
activation3 = Activation_Softmax()

optimizer = Optimizer_SGD()

loss_function = Loss()

params = cp.load("model1-3.npz")

inputs, y_true = (X_test, y_test)

layer1.weights  = params["w1"]
layer1.biases = params["b1"]

layer2.weights  = params["w2"]
layer2.biases = params["b2"]

layer3.weights  = params["w3"]
layer3.biases = params["b3"]

print("En train de tester (...)")
layer1.forward(inputs)
activation1.forward(layer1.output)

layer2.forward(activation1.output)
activation2.forward(layer2.output)

layer3.forward(activation2.output)
activation3.forward(layer3.output)


print("Calcul des pertes (...)")
loss = loss_function.calculate(activation3.output, y_true)
predictions = cp.argmax(activation3.output, axis=1)
accuracy = cp.mean(predictions == y_test)

print("Fin du test")
print(f"precision: {float(accuracy) * 100:.2f} %")
print("Pertes: ", loss)