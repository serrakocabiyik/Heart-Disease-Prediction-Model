#!/usr/bin/env python
# coding: utf-8

# In[2]:


#loading dataset
import pandas as pd
import numpy as np
#visualisation
import matplotlib.pyplot as plt
from sklearn.gaussian_process.kernels import Hyperparameter
%matplotlib inline
import seaborn as sns
#EDA
from collections import Counter
import pandas as pd
df= pd.read_csv("heart.csv")
df

# In[4]:


df.info()

# In[3]:


from sklearn.model_selection import train_test_split
x ,y =df.drop('target',axis=1),df['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state = 9)

# ### Scale-Sensitive

# * Standard Scaler Classifier

# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 

# * K-Neighbors Classifier

# In[11]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train_scaled,y_train)

# * Logistic Regression

# In[12]:


from sklearn.linear_model import LogisticRegression

log = LogisticRegression()
log.fit(x_train_scaled,y_train)

# * Support Vector Classification (SVC)

# In[13]:


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train_scaled,y_train)

# ### SCALE-INSENSİTİVE

# * Random Forest Classifier

# In[14]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(random_state=9)
forest.fit(x_train,y_train)

# * Gradient Boosting Classifier

# In[15]:


from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier()
gb_clf.fit(x_train, y_train)

# * Naive Bayes

# In[16]:


from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(x_train, y_train)

# * ACCURACY

# In[11]:


forest.score(x_test, y_test)

# In[12]:


gb_clf.score(x_test, y_test)

# In[13]:


nb_clf.score(x_test, y_test)

# In[19]:


knn.score(x_test_scaled,y_test)

# In[18]:


log.score(x_test_scaled,y_test)

# In[23]:


svc.score(x_test_scaled,y_test)

# In[17]:



model_ev = pd.DataFrame({
    'Model': ['Random Forest','Gradient Boost','Naive Bayes','KNN','Logistic Regression','SVC'],
    'Accuracy': [
        forest.score(x_test, y_test)*100,
        gb_clf.score(x_test, y_test)*100,
        nb_clf.score(x_test, y_test)*100,
        knn.score(x_test_scaled,y_test)*100,
        log.score(x_test_scaled,y_test)*100,
        svc.score(x_test_scaled, y_test)*100
    ]
})

model_ev


# In[18]:


import matplotlib.pyplot as plt

colors = ['red','green','blue','orange','purple','cyan']

plt.figure(figsize=(12,5))
plt.title("Barplot Representing Accuracy of Different Models")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy %")

bars = plt.bar(model_ev['Model'], model_ev['Accuracy'], color=colors)

# Barların üstüne yüzdeyi yazdırma
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5,
             f"{yval:.2f}%", ha='center', va='bottom', fontsize=10)

plt.show()


# In[19]:


from sklearn.metrics import roc_curve, roc_auc_score

# Random Forest
rf_probs = forest.predict_proba(x_test)[:,1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# Gradient Boost
gb_probs = gb_clf.predict_proba(x_test)[:,1]
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

# Naive Bayes
nb_probs = nb_clf.predict_proba(x_test)[:,1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

# Logistic Regression
lr_probs = log.predict_proba(x_test_scaled)[:,1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# KNN
knn_probs = knn.predict_proba(x_test_scaled)[:,1]
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

# SVC (decision_function çünkü bazı kernel’lerde predict_proba yok)
svc_scores = svc.decision_function(x_test_scaled)
svc_fpr, svc_tpr, _ = roc_curve(y_test, svc_scores)


# Çizim
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('Receiver Operating Characteristic Curve')

plt.plot(rf_fpr, rf_tpr, label='Random Forest')
plt.plot(gb_fpr, gb_tpr, label='Gradient Boosting')
plt.plot(nb_fpr, nb_tpr, label='Naive Bayes')
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression')
plt.plot(knn_fpr, knn_tpr, label='KNN')
plt.plot(svc_fpr, svc_tpr, label='SVC')

# Referans çizgi (rastgele sınıflandırıcı)
plt.plot([0,1], [0,1], linestyle='--', color='gray')

plt.ylabel('True Positive Rate (Recall)')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# In[21]:


roc_auc_score(y_test,rf_probs)

# HYPERPARAMETER TUNING

# In[22]:


from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1 ,2 ,4],
    'max_features': ['sqrt' ,'log2', None] }
forest = RandomForestClassifier(n_jobs=-1 ,random_state=9)
grid_search = GridSearchCV(forest, param_grid, cv=3, n_jobs=-1, verbose=2)

# In[23]:


grid_search.fit(x_train, y_train)

# In[24]:


forest = grid_search.best_estimator_

# In[25]:


forest

# In[26]:


forest.score(x_test, y_test)

# In[27]:


rf_probs = forest.predict_proba(x_test)[:,1]

rf_false_positive_rate, rf_true_positive_rate, rf_threshold = roc_curve(y_test, rf_probs)
# Çizim
sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
plt.title('Receiver Operating Characteristic Curve')

plt.plot(rf_false_positive_rate, rf_true_positive_rate, label='Random Forest')
# Referans çizgileri
plt.plot([0,1],[0,1],'k--')

plt.ylabel('True Positive Rate (Recall)')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()

# FEATURE IMPORTANCES

# In[28]:


feature_importances = forest.feature_importances_
features = forest.feature_names_in_

sorted_idx = np.argsort(feature_importances)
sorted_features = features[sorted_idx]
sorted_importances = feature_importances[sorted_idx]

colors = plt.cm.Blues(np.linspace(0.3,1,len(sorted_features)))

plt.barh(sorted_features, sorted_importances, color=colors)
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importances')

plt.show()

# In[30]:


plt.figure(figsize=(12,10))
sns.heatmap(abs(df.corr()), annot=True, cmap='Blues')

# In[25]:


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Random Forest
rf_probs = forest.predict_proba(x_test)[:,1]
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# Gradient Boost
gb_probs = gb_clf.predict_proba(x_test)[:,1]
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)

# Naive Bayes
nb_probs = nb_clf.predict_proba(x_test)[:,1]
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_probs)

# Logistic Regression
lr_probs = log.predict_proba(x_test_scaled)[:,1]
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)

# KNN
knn_probs = knn.predict_proba(x_test_scaled)[:,1]
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)

# SVC (decision_function çünkü bazı kernel’lerde predict_proba yok)
svc_scores = svc.decision_function(x_test_scaled)
svc_fpr, svc_tpr, _ = roc_curve(y_test, svc_scores)

# Plot
sns.set_style('whitegrid')
plt.figure(figsize=(10,6))
plt.title('Receiver Operating Characteristic Curve (ROC)')

plt.plot(rf_fpr, rf_tpr, label='Random Forest')
plt.plot(gb_fpr, gb_tpr, label='Gradient Boosting')
plt.plot(nb_fpr, nb_tpr, label='Naive Bayes')
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression')
plt.plot(knn_fpr, knn_tpr, label='KNN')
plt.plot(svc_fpr, svc_tpr, label='SVC')

# Referans çizgi (rastgele sınıflandırıcı)
plt.plot([0,1], [0,1], linestyle='--', color='gray')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# FEATURE IMPORTANCES COMPARISON (6 MODEL)

# In[32]:


from sklearn.inspection import permutation_importance

features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal']

# Logistic Regression
log_imp = abs(log.coef_[0])

# SVC (Permutation Importance)
svc_result = permutation_importance(svc, x_test_scaled, y_test, n_repeats=20, random_state=9)
svc_imp = svc_result.importances_mean

# KNN (Permutation Importance)
knn_result = permutation_importance(knn, x_test_scaled, y_test, n_repeats=20, random_state=9)
knn_imp = knn_result.importances_mean

# Random Forest
rf_imp = forest.feature_importances_

# Gradient Boosting
gb_imp = gb_clf.feature_importances_

# Naive Bayes (Permutation Importance ile)
nb_result = permutation_importance(nb_clf, x_test, y_test, n_repeats=20, random_state=9)
nb_imp = nb_result.importances_mean

# Tek DataFrame
data_imp = pd.DataFrame({
    'Feature': features,
    'LogisticRegression': log_imp,
    'SVC': svc_imp,
    'KNN': knn_imp,
    'RandomForest': rf_imp,
    'GradientBoost': gb_imp,
    'NaiveBayes': nb_imp
})

# Grafik (yan yana barlar)
x = np.arange(len(features))  # feature sayısı
width = 0.14  # bar genişliği (6 model için küçültüldü)

plt.figure(figsize=(30,15))
plt.barh(x - 2.5*width, data_imp['LogisticRegression'], height=width, label='LogisticRegression')
plt.barh(x - 1.5*width, data_imp['SVC'], height=width, label='SVC')
plt.barh(x - 0.5*width, data_imp['KNN'], height=width, label='KNN')
plt.barh(x + 0.5*width, data_imp['RandomForest'], height=width, label='RandomForest')
plt.barh(x + 1.5*width, data_imp['GradientBoost'], height=width, label='GradientBoost')
plt.barh(x + 2.5*width, data_imp['NaiveBayes'], height=width, label='NaiveBayes')

plt.yticks(x, data_imp['Feature'])
plt.xlabel("Importance")
plt.title("Feature Importance Comparison (6 Models)")
# Legend’i grafiğin içinde bırak, sadece yazı boyutu büyüsün
plt.legend(loc='upper right', prop={'size':20})
plt.tight_layout()
plt.show()


# 
