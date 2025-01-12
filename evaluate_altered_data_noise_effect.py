from tkinter.constants import ACTIVE

from sklearn.metrics import confusion_matrix,accuracy_score
from  logistic_training_altered_model import training_altered_data

def evaluate_altered_data_with_noise_effect(decision_boundary=6):
     print(f"Evaluating Decision Boundary : X > {decision_boundary}")
     for each_noise_level in [0.1,0.5,0.8]:
         print("\n Noise Level:" ,{each_noise_level})
         model,X,Y=training_altered_data(decision_boundary, each_noise_level)

         Y_pred=model.predict(X)
         accuracy=accuracy_score(Y,Y_pred)
         confusionmatrix=confusion_matrix(Y,Y_pred)
         print(f"Accuracy is {accuracy:.2f}")
         print(f"Confusion matrix is {confusionmatrix}")

if __name__=="__main__":
    evaluate_altered_data_with_noise_effect(decision_boundary=6)






