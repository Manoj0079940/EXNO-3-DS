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
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df
![image](https://github.com/user-attachments/assets/322a9402-3c26-44cf-bedb-fb9a6593cd62)
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
![image](https://github.com/user-attachments/assets/2a8c1047-a756-4963-80a2-700ece345eed)
df['bo2']=e1.fit_transform(df[["ord_2"]])
![image](https://github.com/user-attachments/assets/66e5bbe4-ff06-4871-8691-aba7cdfa9911)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
![image](https://github.com/user-attachments/assets/301461da-96c3-4252-bbbf-45ef96d955a0)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)  # Change 'sparse' to 'sparse_output'
df2 = df.copy()
enc = pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2 = pd.concat([df2, enc], axis=1)
df2
![image](https://github.com/user-attachments/assets/b6d72fdb-b3b2-405c-9150-38fd43c0174c)
pip install --upgrade category_encoders
![image](https://github.com/user-attachments/assets/8abc1fe9-e971-4a6e-9bfe-649cc1fa64d8)
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
![image](https://github.com/user-attachments/assets/b92c690d-2291-419a-bf7c-8c5014afdb3b)
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
![image](https://github.com/user-attachments/assets/50e61a61-7f6f-40cb-97d9-7b16ea350bfe)
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
![image](https://github.com/user-attachments/assets/7ff7171e-0d3b-42d2-853a-44f22fcda1d2)
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
![image](https://github.com/user-attachments/assets/c71c59a2-310e-45db-8ce8-63f6acfdb107)
np.log(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/f120beac-11a3-47b0-bff5-ca471c28552b)
np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/user-attachments/assets/5429c323-7bc7-4778-a842-d674f47f37ac)
np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/b7d6b896-706a-493e-bbb6-3ef8c9a65575)
np.square(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/e4af2811-e040-4e07-986c-ad65eee0ed60)
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/f9c4af62-d42b-4fd2-b452-f873dd841b16)
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
![image](https://github.com/user-attachments/assets/180e8126-6e81-497a-bf86-2e95742578eb)
df.skew()
![image](https://github.com/user-attachments/assets/cb388047-2f9b-4bee-97f0-471b4f032ea1)
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
![image](https://github.com/user-attachments/assets/23495094-3c19-452d-9ac6-126492f08f4a)
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
![image](https://github.com/user-attachments/assets/61e1b4e8-cb29-4b03-a625-3c8e819e577c)
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line="45")
plt.show()
![image](https://github.com/user-attachments/assets/b82889d1-6cd3-4c66-868e-d34eebb3a0b1)
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line="45")
plt.show()
![image](https://github.com/user-attachments/assets/faa38692-0e33-47ab-a0af-bf12e67cf73b)
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew_2"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew_2"],line="45")
plt.show()
![image](https://github.com/user-attachments/assets/c1b7946a-a024-4087-b504-54fad89bc844)
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line="45")
plt.show()
![image](https://github.com/user-attachments/assets/7c4cdd26-b766-4bca-a38d-6a85e40c0a03)
sm.qqplot(df["Highly Negative Skew_1"],line="45")
plt.show()
![image](https://github.com/user-attachments/assets/a861edf0-e030-4e4e-9441-f94dceb402d2)
dt=pd.read_csv("titanic_dataset.csv")
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt["Age"],line="45")
plt.show()
![image](https://github.com/user-attachments/assets/990307f0-5a50-485d-a992-7f60f6646da5)
sm.qqplot(dt['Age_1'],line="45")
plt.show()
![image](https://github.com/user-attachments/assets/56067611-8def-48b8-a08b-316151e604b4)

# RESULT:
     Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
