from sklearn.linear_model import LogisticRegression
from synthetic_data_simulation import simulate_classification_data_altered

def training_altered_data(decision_boundary=6,noise_level=0.1):
      X,Y = simulate_classification_data_altered(decision_boundary,noise_level)
      model = LogisticRegression()
      model.fit(X,Y)
      return model,X,Y

if __name__ == '__main__':
    model, X, Y = training_altered_data(decision_boundary=6,noise_level=0.1)
    print(f"Model is trained successfully")
    print(f"Model Intercept(b):" ,model.intercept_)
    print(f"Model Coefficient(w):", model.coef_)

