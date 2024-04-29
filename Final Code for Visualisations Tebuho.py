import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import scipy.stats as stats

# Load the processed customer dataset
file_path = 'preprocessed_customer_shopping_data.csv'
df = pd.read_csv(file_path)

print("This is Visualisation 18") #Visualisation 18
# Encode categorical variables
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['payment_method'] = df['payment_method'].map({'Credit Card': 0, 'Cash': 1})
 
# Define variables for correlation analysis
variables = ['age', 'quantity', 'gender', 'price', 'payment_method']

# Calculate the correlation matrix
corr_matrix = df[variables].corr()
print(corr_matrix)
# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis')
plt.title('Correlation Matrix Heatmap')
plt.savefig('heatmap_of_variable_correlations.png')
plt.show()

print("This is Visualisation 3") #Visualisation 3
# Aggregate data for heatmap
monthly_category_spend = df.groupby(['month', 'category'])['price'].sum().unstack()

# Heatmap for monthly spend for each year and category
plt.figure(figsize=(12, 8))
sns.heatmap(monthly_category_spend, annot=True, fmt='.0f', cmap='viridis', linewidths=.5)
plt.title('Heatmap of Monthly Spend by Product Category')
plt.ylabel('Month')
plt.xlabel('Product Category')
plt.tight_layout()
plt.savefig('monthly_spend_heatmap.png')
plt.show()

print("This is Visualisation 14") #Visualisation 14
# Scatter plot for customer segmentation using price(transaction total spend per customer), age, and gender
plt.figure(figsize=(12, 8), facecolor='white')
sns.scatterplot(data=df, x='age', y='price', hue='gender', style='gender', s=100)

plt.title('Customer Segmentation by Age, Gender, and Price')
plt.xlabel('Age')
plt.ylabel('Price')
plt.grid(True)
plt.savefig('customer_segmentation.png')
plt.show()

print ("This is Visualisation 8") #Visualisation 8
data = pd.read_csv('preprocessed_customer_shopping_data.csv')
# Group by 'shopping_mall' and 'payment_method' and sum the 'quantity' column
quantity_totals_by_mall = data.groupby(['shopping_mall', 'payment_method'])['quantity'].sum().unstack()

# Calculate the maximum y-value rounded up to the nearest multiple of 5,000, plot in increments of 5,000
max_y_value = quantity_totals_by_mall.max().sum()
max_tick = (max_y_value // 5000 + 1) * 5000

# Stacked bar chart for sum of quantity column by payment method at each mall
plt.figure(figsize=(14, 10))
ax = quantity_totals_by_mall.plot(kind='bar', stacked=True, colormap='viridis', alpha=0.75)
plt.title('Total Quantity Sold by Payment Method at Each Mall', fontsize=12)
plt.xlabel('Shopping Mall', fontsize=14)
plt.ylabel('Total Quantity Sold', fontsize=14)
plt.xticks(rotation=90, fontsize=12, ha='center')
plt.yticks(range(0, max_tick, 5000), fontsize=12)
plt.legend(title='Payment Method', fontsize=12)
plt.tight_layout()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('quantity_totals_by_mall.png')
plt.show()

print ("This is the correlation and significance analysis") #Correlation and significance analysis
# Dataframe with payment methods as dummy variables and the quantity column
df_payment_method = pd.get_dummies(data['payment_method'])
df_payment_method['quantity'] = data['quantity']

# Correlation and p-values for Cash
corr_cash, p_value_cash = stats.pearsonr(df_payment_method['Cash'], df_payment_method['quantity'])

# Correlation and p-values for Credit Card
corr_credit_card, p_value_credit_card = stats.pearsonr(df_payment_method['Credit Card'], df_payment_method['quantity'])

# Correlation and p-values for Debit Card
corr_debit_card, p_value_debit_card = stats.pearsonr(df_payment_method['Debit Card'], df_payment_method['quantity'])

print('Cash Correlation:', corr_cash, 'P-value:', p_value_cash)
print('Credit Card Correlation:', corr_credit_card, 'P-value:', p_value_credit_card)
print('Debit Card Correlation:', corr_debit_card, 'P-value:', p_value_debit_card)

