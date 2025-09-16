import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("housing.csv" , delim_whitespace=True)
df.columns = [
    "CRIM",      # Crime rate per capita by town
    "ZN",        # Proportion of residential land zoned for lots over 25,000 sq. ft.
    "INDUS",     # Non-retail business acres proportion
    "CHAS",      # Charles River dummy variable (1 if near river, else 0)
    "NOX",       # Nitric oxide concentration
    "RM",        # Average number of rooms per dwelling
    "AGE",       # % of owner-occupied units built before 1940
    "DIS",       # Distance to employment centers
    "RAD",       # Accessibility to radial highways
    "TAX",       # Property tax rate per $10,000
    "PTRATIO",   # Pupil-teacher ratio by town
    "B",         # 1000(Bk - 0.63)^2 (Bk = proportion of Black residents)
    "LSTAT",     # % lower status of the population
    "MEDV"       # Median value of owner-occupied homes ($1000s) [TARGET]
]

# print(df.head())

x = df.drop("MEDV" , axis = 1)
y = df["MEDV"]

x_train , x_test , y_train , y_split = train_test_split(
    x , y , test_size=0.2 , random_state=42
    )

# test_size=0.2 → 20% of data goes to testing set.
# random_state=42 → ensures reproducibility (you’ll get the same split every time).
# The function returns four datasets:
# X_train: Features for training
# X_test: Features for testing
# y_train: Target values for training
# y_test: Target values for testing

# print("Training set size:", x_train.shape)
# print("Testing set size:", x_test.shape)


# train the model 
model = LinearRegression()
model.fit(x_train , y_train)

# predictions 
y_pred = model.predict(x_test)

# shows average error in predictions
mae = mean_absolute_error(y_split, y_pred)
print("Mean Absolute Error:", mae)

# measures how well model fits the data 
r2 = r2_score(y_split , y_pred)
print(f"R2 score : {r2}")

# which features affect the pricing 
coefficients = pd.DataFrame({
    "Feature" : x.columns ,
    "Coefficient" : model.coef_
})

print(coefficients.sort_values(by = "Coefficient" , ascending=False))


plt.figure(figsize=(8, 6))

# Scatter plot of actual vs predicted
plt.scatter(y_split, y_pred, color='blue', alpha=0.6, label="Predicted Points")

# Perfect prediction line
plt.plot([y_split.min(), y_split.max()], 
         [y_split.min(), y_split.max()], 
         color='red', linewidth=2, label="Perfect Prediction")

# Labels and title
plt.xlabel("Actual House Prices (MEDV)")
plt.ylabel("Predicted House Prices (MEDV)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)

plt.show()