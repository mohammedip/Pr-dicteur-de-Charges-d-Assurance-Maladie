from data import df , sns ,plt , pd , np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#delete duplicates
df_new = df.drop_duplicates().copy()

# Q1 = df_new['bmi'].quantile(0.25)

# Q3 = df_new['bmi'].quantile(0.75)

# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# df_no_outliers = df_new[(df_new["bmi"] >= lower) & (df_new["bmi"] <= upper)]

#log charges
df_new["log_charges"] = np.log(df_new["charges"])

#One-hot encoder 
df_encoded = pd.get_dummies(df_new, columns=["sex", "smoker", "region"], dtype=int)

df_encoded = df_encoded.drop("charges", axis=1)

#Standarisation
num_cols = ["age", "bmi", "children"]

scaler = StandardScaler()
df_scaled = df_encoded.copy()
df_scaled[num_cols] = scaler.fit_transform(df_encoded[num_cols])


#Split data 20/80
X = df_scaled.drop("log_charges", axis=1)   
y = df_scaled["log_charges"]            

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      
    random_state=42,     
    shuffle=True       
)



