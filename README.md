# data-preprocessing
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.impute import SimpleImputer
import numpy as np

# Initialize the SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer to the data and transform it
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Encoding the Independent Variable
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Apply OneHotEncoder to the first column
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), [0])
    ],
    remainder='passthrough'
)

X = ct.fit_transform(X)

# Optional: Scaling features (if needed)
scaler = StandardScaler()
X = scaler.fit_transform(X)

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

