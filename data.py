from sklearn.datasets import fetch_openml
import cupy as cp
import matplotlib.pyplot as plt


mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target

X = X.astype("float32")/ 255.0
y = y.astype("int32")

X = cp.asarray(X)
y = cp.asarray(y)

X_train = X[0:60000]
y_train = y[0:60000]

X_test = X[60000:70000]
y_test = y[60000:70000]

X_val = X_train[55000:60000]
y_val = y_train[55000:60000]

X_train = X_train[0:55000]
y_train = y_train[0:55000]

if __name__ == "__main__":
    image = cp.asnumpy(X_train[1]).reshape(28, 28)
    plt.imshow(image, cmap="gray")
    plt.title(f"label: {int(y_train[1].item())}")
    plt.axis("off")
    plt.show()