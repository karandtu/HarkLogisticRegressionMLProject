#evaluating the model using accuracy and confusion matrix

from sklearn.metrics import confusion_matrix,accuracy_score
from logistic_training import train_logistic_data

def evaluate_logistic_model():
     model,X,Y = train_logistic_data()

     Y_pred=model.predict(X)
     accuracy=accuracy_score(Y,Y_pred)
     confusionMatrix = confusion_matrix(Y,Y_pred)
     print(f"Accuracy: {accuracy:.2%}")
     print(f"Accuracy: Accuracy",accuracy)
     print(f"Confusion Matrix: {confusionMatrix}")
     print(f"Confusion Matrix:\n", confusionMatrix)

if __name__=="__main__":
    evaluate_logistic_model()

