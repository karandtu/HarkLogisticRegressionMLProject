#Hands-on implementation project that classifies
#whether a student passes or fails based on study hours

#In Logistic Regression we use sigmoid function to map predictions
# to create probabilities between binary outcomes 0 and 1.

#In linear regression we conduct several tests/different ways
# to test new forms of all data to make sure our 1-and-only single
#outcome should come i.e. single value/linear outcome.


import numpy as np
import matplotlib.pyplot as plt

def simulate_classification_data():
    np.random.seed(42)
    X=np.random.rand(100,1)*10
    Y=(X>5).astype(int)  # if true/false, convert boolean array into int 0 or 1
    return X,Y

#add some noise
    #y=np.where(condition if true, then y flips to 1-y, otherwise keep y as y)
    Y = np.where(np.random.rand(100, 1) > 0.9, 1 - Y, Y)

#scatterplots
    plt.scatter(X,Y,label="Data Points",c=Y,cmap="bwr")
    plt.axvline(X=5,color="gray",linestyle="--",label="Decision Boundary")
    plt.xlabel("Study Hours")
    plt.ylabel("Pass (1)/ Fail (0)")
    plt.title("Data Simulation with Logistic Regression")
    plt.legend()
    plt.show()

if __name__=="__main__":
    simulate_classification_data()





