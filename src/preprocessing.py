# Import additional libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Step 0: Define feature types FIRST (before using them)
categorical_features = ['Gender', 'ProductCategory', 'LoyaltyProgram']

numerical_features = [
    'Age',
    'AnnualIncome',
    'NumberOfPurchases',
    'TimeSpentOnWebsite',
    'DiscountsAvailed'
]

# Step 1: Check if missing values exist and handle them
# Handle missing values for numerical features using median
for col in numerical_features:
    df[col].fillna(df[col].median(), inplace=True)

# Handle missing values for categorical features using mode
for col in categorical_features:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Step 2: Create a new feature AvgSpendingPerPurchase
df['AvgSpendingPerPurchase'] = (
    df['AnnualIncome'] / df['NumberOfPurchases'].replace(0, 1)
)

# Step 3: Append AvgSpendingPerPurchase to the list of numerical features
numerical_features.append('AvgSpendingPerPurchase')

# Step 4: Create preprocessor with pipelines for categorical and numerical features
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Step 5: Split the Dataset as outlined in the instructional steps above
# Split the data into features and target
X = df.drop('PurchaseStatus', axis=1)
y = df['PurchaseStatus']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 6: Fit and Transform
# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)
