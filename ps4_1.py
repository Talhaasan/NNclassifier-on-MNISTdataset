import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from mlxtend.plotting import plot_learning_curves


x, y = fetch_openml('mnist_784',return_X_y=True)
x = x / 255.0

x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]

mlp = MLPClassifier(hidden_layer_sizes=(20,30),learning_rate_init=0.1)

mlp.fit(x_train, y_train)

print("Training set score: %f" % mlp.score(x_train, y_train))
print("Test set score: %f" % mlp.score(x_test, y_test))

loss_values = mlp.loss_curve_
plt.title("Loss")
plt.plot(loss_values)
plt.legend()
plt.show()

plt.title("Learning Curve")
plot_learning_curves(x_train, y_train, x_test, y_test, mlp)
plt.show()
