import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


with open('Predictions/pickle_Pred_Quran_Bukhari_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)


print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))








with open('Predictions/pickle_Pred_Reuter_C50_All_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)

y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


with open('Predictions/pickle_Pred_Random_Reuter_C50_All_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)

y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


with open('Predictions/pickle_Pred_Spooky_Author_All_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)


y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


with open('Predictions/pickle_Pred_Spooky_Author_All_Random_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)


y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)

print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))



with open('Predictions/pickle_Pred_Spooky_Author_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)


print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))


with open('Predictions/pickle_Pred_Spooky_Author_Random_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)


print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))



with open('Predictions/pickle_Pred_New_Sketch_1.pickle','rb') as f:
    y_test, y_pred =pickle.load(f)


print(accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
