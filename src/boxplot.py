from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

#import matplotlib for plotting
import matplotlib.pyplot as plt

#create a boxplot for the 'MedHouseVal' column
df.boxplot(column='MedHouseVal') #this will create a boxplot for the 'MedHouseVal' column
plt.title('Boxplot of Cali Median House Value')
plt.ylabel('Median House Value')
plt.tight_layout()

# you can save the boxplot...
plt.savefig("figs/boxplot.png")

print("Boxplot saved to figs/boxplot.png")