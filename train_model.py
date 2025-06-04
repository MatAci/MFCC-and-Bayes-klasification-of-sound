from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import joblib

data = np.load('features_labels_segment.npz')
X = data['X']
y = data['y']

# 1) Podijeli podatke
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2) Kreiraj i treniraj Naive Bayes (gaussov)
clf = GaussianNB()
clf.fit(X_train, y_train)

joblib.dump(clf, 'naive_bayes_model1.pkl')
print("Model spremljen kao 'naive_bayes_model1.pkl'")

# 3) Predikcija na test skupu
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1] # Vjerojatnosti za pozitivnu klasu (Voiced)

# 4) Procjena performansi
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Unvoiced','Voiced']))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 5) ROC krivulja i AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

# Crtanje ROC krivulje
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

