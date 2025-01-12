import numpy as np
import matplotlib.pyplot as plt
from logistic_training import train_logistic_data


def plot_predictions():
    model ,X, Y = train_logistic_data()

#Predict results on new data points
    X_new= np.linspace(0,10,100).reshape(-1,1)
    Y_pred= model.predict_proba(X_new)[:,1]

    plt.scatter(X, Y,label="Data Points", c=Y, cmap="bwr")
    plt.plot(X_new, Y_pred, color="green", label="Sigmoid Curve")
    plt.axvline(X=5, color="gray",linestyle="--",label="Decision Boundary")
    plt.xlabel("Study Hours")
    plt.ylabel("Probability of passing")
    plt.title("Logistic Regression Prediction Possibilities")
    plt.legend()
    plt.show()

if __name__ == "__main__":
     plot_predictions()


