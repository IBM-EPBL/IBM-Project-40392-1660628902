import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv(r'C:\Users\DELL\Desktop\heart disease/Heart_Disease_Prediction.csv')
data.head()
data.info()
data.describe(include = 'all')
data.isnull().sum()
data.nunique()
data.columns
colm = ['Sex', 'Chest pain type','FBS over 120','EKG results','Exercise angina','Slope of ST', 'Number of vessels fluro', 'Thallium', 'Heart Disease']
for col in colm:
  sns.countplot(data[col])
  plt.show()
  plt.figure(figsize=(12,10))
corr = data.corr()
sns.heatmap(corr, annot = True, linewidths= 0.2, linecolor= 'black', cmap = 'afmhot')
data.columns
X = data[['Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 'FBS over 120',
       'EKG results', 'Max HR', 'Exercise angina', 'ST depression',
       'Slope of ST', 'Number of vessels fluro', 'Thallium']]
y = data['Heart Disease']
print(X.shape,y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42529)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
train_convert = {"Absence":0,"Presence":1}
y_train = y_train.replace(train_convert)
test_convert = {"Absence":0,"Presence":1}
y_test = y_test.replace(test_convert)
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.fit_transform(X_test)
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
cm = confusion_matrix(y_test,pred)
print(classification_report(y_test,pred))
sns.heatmap(cm, annot = True, fmt = 'g', cbar = False, cmap = 'icefire', linewidths= 0.5, linecolor= 'grey')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
print("Accuracy Score = {}".format(round(accuracy_score(y_test,pred),5)))
