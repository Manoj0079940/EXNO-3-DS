
## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/7054cc26-0527-4f2b-88be-cee096a15f08)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/86bdc3b1-0272-4332-8ecd-6d03bc3be6e9)
### df['bo2']=e1.fit_transform(df[["ord_2"]])
![image](https://github.com/user-attachments/assets/ad1998b5-6399-49eb-8470-cb8b352870cd)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/26b44bb6-48de-45d2-9b8e-294d64bf6739)
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  # Change 'sparse' to 'sparse_output'
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2, enc], axis=1)
df2
```
![image](https://github.com/user-attachments/assets/45eca4c1-0059-47f5-9323-fd1b6d5b46a8)
### pip install --upgrade category_encoders
![image](https://github.com/user-attachments/assets/3a0b5511-cb84-4296-b750-b0f79f4d809f)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
```
![image](https://github.com/user-attachments/assets/d7445846-e9d9-4148-b889-2e6df35f9736)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/95156109-87c7-439c-8815-1d30041bb4a0)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```
![image](https://github.com/user-attachments/assets/6715fc51-7bc5-4481-aa84-b4c3824b0c48)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/f11d9f42-607a-422b-8bed-b3b854eb6cc9)
### np.log(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/d58b7d63-e566-4d5f-b132-cb84b3beb070)
### np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/user-attachments/assets/7b3e31f8-2851-41da-b914-641ccc6d625f)
### np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/9498e899-d0a2-4156-af72-7d7d7d972174)
### np.square(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/c4aad705-2cd7-4143-8780-415e998cb80d)
### df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
#![image](https://github.com/user-attachments/assets/51c2898b-8b33-4cba-9b8e-d71b976c5985)
### df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
![image](https://github.com/user-attachments/assets/b5b4055d-0b48-46d0-b4b8-bc787a15ec26)
### df.skew()
![image](https://github.com/user-attachments/assets/ecc4965b-9c6a-4787-a130-556ee8e44ecf)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/0bf13e5b-655c-4f86-93a2-da460655e361)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/c6b20c75-390f-4ff3-8be9-f61ebdb348de)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/26a115b4-8c95-44a5-a16c-68ea362d0100)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/1e357079-ccfc-444d-812e-9d833f880d36)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew_2"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew_2"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/8b5e48c9-f194-45ba-b259-6c9bdcaf69d2)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/04654288-ea39-4c47-8167-6f62d5775e86)
```
sm.qqplot(df["Highly Negative Skew_1"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/7949fd81-db8e-40b1-ac42-1e074290fe1a)
```
dt=pd.read_csv("titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/543c0a4a-2ca1-47dd-be29-f6b2a0e39786)
```
sm.qqplot(dt['Age_1'],line="45")
plt.show()
```
![image](https://github.com/user-attachments/assets/1ab92eb4-7f36-4d9d-a7a1-35604f74b865)

# RESULT:
     Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
