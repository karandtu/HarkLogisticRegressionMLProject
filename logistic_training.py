from matplotlib.cbook import simple_linear_interpolation
from sklearn.linear_model import LogisticRegression
from data_simulation_logistic import simulate_classification_data

def train_logistic_data():
     X,Y=simulate_classification_data()
     model=LogisticRegression()
     model.fit(X,Y)
     return model,X,Y

if __name__=='__main__':
    model,X,Y=train_logistic_data()
    print(f"Model is trained successfully")
    print(f"Intercept(b for bias):", model.intercept_)
    print(f"Cofficient(w for weight):", model.coef_)





