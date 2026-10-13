
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from data import X_test, y_test
from training import Layer_dense, Activation_ReLu, Activation_Softmax

layer1 = Layer_dense(784, 286)
layer2 = Layer_dense(286, 286)
layer3 = Layer_dense(286, 10)

relu1 = Activation_ReLu()
relu2 = Activation_ReLu()
softmax = Activation_Softmax()


def load_model():
    params = cp.load("model.npz")
    layer1.weights, layer1.biases = params["w1"], params["b1"]
    layer2.weights, layer2.biases = params["w2"], params["b2"]
    layer3.weights, layer3.biases = params["w3"], params["b3"]


def predict(image):
    x = cp.asarray(image).reshape(1, -1)

    layer1.forward(x)
    relu1.forward(layer1.output)

    layer2.forward(relu1.output)
    relu2.forward(layer2.output)

    layer3.forward(relu2.output)
    softmax.forward(layer3.output)

    return cp.asnumpy(softmax.output[0])


def main():
    load_model()

    images = cp.asnumpy(cp.asarray(X_test))
    labels = cp.asnumpy(cp.asarray(y_test)).reshape(-1).astype(int)

    print("Running predictions on the test dataset...")

    probabilities = np.array([predict(img) for img in images])
    predictions = np.argmax(probabilities, axis=1)
    accuracy = np.mean(predictions == labels) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.25)

    image_artist = ax.imshow(
        images[0].reshape(28, 28),
        cmap="gray",
        interpolation="nearest",
    )
    ax.axis("off")

    def update_image(index):
        i = int(index)
        image_artist.set_data(images[i].reshape(28, 28))

        predicted = int(predictions[i])
        actual = int(labels[i])
        confidence = probabilities[i, predicted] * 100

        correct = predicted == actual
        color = "green" if correct else "red"
        status = "CORRECT" if correct else "INCORRECT"

        ax.set_title(
            f"Image {i + 1} / {len(images)}\n"
            f"Actual digit: {actual} | Prediction: {predicted}\n"
            f"Confidence: {confidence:.2f}%\n"
            f"Prediction: {status}",
            color=color,
            fontsize=13,
        )
        fig.canvas.draw_idle()

    slider_ax = fig.add_axes([0.18, 0.12, 0.64, 0.035])
    slider = Slider(
        slider_ax,
        "Image",
        0,
        len(images) - 1,
        valinit=0,
        valstep=1,
    )
    slider.on_changed(update_image)

    # Previous / Next buttons
    previous_ax = fig.add_axes([0.25, 0.035, 0.18, 0.055])
    next_ax = fig.add_axes([0.57, 0.035, 0.18, 0.055])

    previous_button = Button(previous_ax, "< Previous")
    next_button = Button(next_ax, "Next >")

    previous_button.on_clicked(
        lambda event: slider.set_val(max(0, int(slider.val) - 1))
    )
    next_button.on_clicked(
        lambda event: slider.set_val(
            min(len(images) - 1, int(slider.val) + 1)
        )
    )

    update_image(0)

    print(f"Test accuracy: {accuracy:.2f}%")
    print("Use the slider or buttons to browse the test images.")

    plt.show()


if __name__ == "__main__":
    main()
