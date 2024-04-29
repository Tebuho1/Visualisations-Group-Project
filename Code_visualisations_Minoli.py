import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import scipy.stats as stats

#We tried and ran all of the following codes on Jupyter

#Figure 1 - Spark outputs 
# Importing the necessary libraries
from pyspark.sql import SparkSession

# Creating the SparkSession
spark = SparkSession.builder \
    .appName("Customer Shopping Data Analysis") \
    .getOrCreate()

# Loading the dataset into a Spark DataFrame
df = spark.read.csv('preprocessed_customer_shopping_data.csv', header=True, inferSchema=True)

# Getting the schema as a StructType object
schema = df.schema

# Creating an empty list to store schema information
schema_info = []

# Iterating over each field in the schema and extracting the needed information
for field in schema.fields:
    column_name = field.name
    data_type = str(field.dataType)
    schema_info.append((column_name, data_type))

# Converting the list of tuples into a DataFrame
schema_df = spark.createDataFrame(schema_info, ["Column Name", "Data Type"])

# Showing the schema in tabular format
schema_df.show(truncate=False)

# Stopping the SparkSession
spark.stop()


#Table 3 - Visualisation of statistics 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the CSV file
data = pd.read_csv("preprocessed_customer_shopping_data.csv")

# Calculating the statistics for price, quantity, and age
price_stats = data['price'].describe()
quantity_stats = data['quantity'].describe()
age_stats = data['age'].describe()

# Calculating the percentiles for price, quantity, and age
price_percentiles = np.percentile(data['price'], [25, 50, 75])
quantity_percentiles = np.percentile(data['quantity'], [25, 50, 75])
age_percentiles = np.percentile(data['age'], [25, 50, 75])

# Rounding the statistics to two decimal places
price_stats = price_stats.round(2)
quantity_stats = quantity_stats.round(2)
age_stats = age_stats.round(2)
price_percentiles = np.round(price_percentiles, 2)
quantity_percentiles = np.round(quantity_percentiles, 2)
age_percentiles = np.round(age_percentiles, 2)

# Creating a table using Matplotlib
fig, ax = plt.subplots(figsize=(10, 6))

# Hiding axes
ax.axis('off')

# Creating the table
table_data = [
    ["", "Price", "Quantity", "Age"],
    ["Mean", f"{price_stats['mean']:.2f}", f"{quantity_stats['mean']:.2f}", f"{age_stats['mean']:.2f}"],
    ["SD", f"{price_stats['std']:.2f}", f"{quantity_stats['std']:.2f}", f"{age_stats['std']:.2f}"],
    ["Max", f"{price_stats['max']:.2f}", f"{quantity_stats['max']:.2f}", f"{age_stats['max']:.2f}"],
    ["Min", f"{price_stats['min']:.2f}", f"{quantity_stats['min']:.2f}", f"{age_stats['min']:.2f}"],
    ["25th percentile", f"{price_percentiles[0]:.2f}", f"{quantity_percentiles[0]:.2f}", f"{age_percentiles[0]:.2f}"],
    ["50th percentile", f"{price_percentiles[1]:.2f}", f"{quantity_percentiles[1]:.2f}", f"{age_percentiles[1]:.2f}"],
    ["75th percentile", f"{price_percentiles[2]:.2f}", f"{quantity_percentiles[2]:.2f}", f"{age_percentiles[2]:.2f}"]
]

# Selecting the colours
table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                 cellColours=[['lightblue'] * 4] + [['lightgrey'] * 4] * 7,
                 colColours=['lightgrey'] * 4,
                 bbox=[0, 0, 1, 1])

# Formatting cells
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(fontweight='bold')
    elif j != 0:
        cell.set_text_props(fontweight='bold', color='black')

# Adjusting the font size
table.auto_set_font_size(False)
table.set_fontsize(12)

plt.show()


#Figure 3 - Radar
import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np 


# Loading the dataset 
data = pd.read_csv("preprocessed_customer_shopping_data.csv") 


# Calculating the total number of transactions for each category 
category_counts = data['category'].value_counts() 


# Defining categories and the corresponding counts 
categories = category_counts.index.tolist() 
counts = category_counts.values.tolist() 


# Creating the angles for radar chart 
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist() 


# Making the plot close loop 
counts = np.concatenate((counts,[counts[0]])) 
angles += angles[:1] 


# Creating the spider plot 
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True)) 
ax.fill(angles, counts, color='blue', alpha=0.25)  # Filling the area under the curve 
ax.plot(angles, counts, color='blue', linewidth=2)  # Plotting the lines 


# Customising the spider chart 
ax.set_yticklabels([])  # Hide radial axis labels 
ax.set_xticks(angles[:-1])  # Set radial ticks 
ax.set_xticklabels(categories)  # Set category labels 
ax.set_title('Popularity of Categories')  # Set the title 

 
# Display the spider plot 
plt.show() 


#Figure 4
import pandas as pd 
import matplotlib.pyplot as plt 

 

# Loading the dataset 
data = pd.read_csv("preprocessed_customer_shopping_data.csv") 

 

# Calculating the average price for each category 
average_prices = data.groupby('category')['price'].mean().sort_values() 

 

# Creating the horizontal bar plot 
plt.figure(figsize=(10, 6)) 

 

# Plotting bars with determined colors 
bars = average_prices.plot(kind='barh', color='lightblue') 

 

# Finding the tallest and shortest bars 
tallest_bar_index = average_prices.idxmax() 
shortest_bar_index = average_prices.idxmin() 

 

# Set the tallest and shortest bars to dark blue to better visualize the differences 
bars.patches[average_prices.index.get_loc(tallest_bar_index)].set_facecolor('darkblue') 
bars.patches[average_prices.index.get_loc(shortest_bar_index)].set_facecolor('darkblue') 

 

plt.title('Average Price for Each Category') 
plt.xlabel('Average Price in Turkish liras (₺)') 
plt.ylabel('Category') 

 

# Set x-axis limit 
plt.xlim(0, 3500) 

 

# Adding the corresponding average price next to each bar in Turkish Lira 
for i, v in enumerate(average_prices): 

    plt.text(v + 50, i, f"₺{v:.2f}", ha='left', va='center')  # Update the text to display prices in Turkish Lira 
    

plt.tight_layout() 
plt.show() 


#Figure 6
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

 

# Loading the dataset 
data = pd.read_csv("preprocessed_customer_shopping_data.csv") 

 

# Defining the ranges for the age groups 
age_groups = { 

    '18-24': range(18, 25), 

    '25-34': range(25, 35), 

    '35-44': range(35, 45), 

    '45-54': range(45, 55), 

    '55+': range(55, 70)   

} 

 

# Mapping the ages to the age groups 
data['age_group'] = data['age'].apply(lambda x: next((group for group, ages in age_groups.items() if x in ages), None)) 

 

# Creating a clustered grouped bar plot 
plt.figure(figsize=(12, 8)) 

 

# Plotting clustered groupd bar plot 
sns.countplot(x='age_group', hue='category', data=data, palette='pastel') 

 

# Setting the title and labels 
plt.title('Relationship between Age Group and Category') 
plt.xlabel('Age Group') 
plt.ylabel('Count') 

 

# Rotating x-axis labels to increase readability  
plt.xticks(rotation=45, ha='right') 

 

# Adding the legend outside the plot  
plt.legend(title='Category', bbox_to_anchor=(1, 1)) 

 

# Adjusting the layout to prevent overlap 
plt.tight_layout() 

 

# Showing the plot 
plt.show() 


#Figure 18
import pandas as pd 
import matplotlib.pyplot as plt 

 

# Loading the dataset 
df = pd.read_csv("preprocessed_customer_shopping_data.csv") 

 

# Converting 'invoice_date' column to datetime 
df['invoice_date'] = pd.to_datetime(df['invoice_date']) 

 

# Filter the dataset to include only dates in the years 2021, 2022, and 2023 
start_date_2021 = pd.to_datetime('2021-01-01') 
end_date_2021 = pd.to_datetime('2021-12-31') 
start_date_2022 = pd.to_datetime('2022-01-01') 
end_date_2022 = pd.to_datetime('2022-12-31') 
start_date_2023 = pd.to_datetime('2023-01-01') 
end_date_2023 = pd.to_datetime('2023-12-31') 

 

df_2021 = df[(df['invoice_date'] >= start_date_2021) & (df['invoice_date'] <= end_date_2021)] 
df_2022 = df[(df['invoice_date'] >= start_date_2022) & (df['invoice_date'] <= end_date_2022)] 
df_2023 = df[(df['invoice_date'] >= start_date_2023) & (df['invoice_date'] <= end_date_2023)] 

 

# Grouping by month and counting the number of invoices for each month in each year 
invoices_per_month_2021 = df_2021.groupby(df_2021['invoice_date'].dt.month).size() 
invoices_per_month_2022 = df_2022.groupby(df_2022['invoice_date'].dt.month).size() 
invoices_per_month_2023 = df_2023.groupby(df_2023['invoice_date'].dt.month).size() 

 

# Plotting the main line graph 
fig, ax = plt.subplots(figsize=(10, 6)) 

 

ax.plot(invoices_per_month_2021.index, invoices_per_month_2021, marker='o', linestyle='-', color='blue', label='2021') 
ax.plot(invoices_per_month_2022.index, invoices_per_month_2022, marker='o', linestyle='-', color='green', label='2022') 
ax.plot(invoices_per_month_2023.index, invoices_per_month_2023, marker='o', linestyle='-', color='orange', label='2023') 

 

# Customising the dot for the 7th data point in 2021 to be red and slightly bigger than the rest 
ax.plot(7, invoices_per_month_2021[7], marker='o', markersize=8, linestyle='-', color='red', label='Max Sales\n(July 2021)') 

 

ax.set_title('Number of Invoices per Month (2021-2023)') 
ax.set_xlabel('Month') 
ax.set_ylabel('Number of Invoices') 
ax.set_xticks(range(1, 13)) 
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 
ax.legend(loc='lower left', fontsize='small')  # Adjust the location and size of the legend 
ax.grid(True) 

 

# Creating a subplot for the pie chart 
ax_pie = fig.add_axes([0.6, 0.15, 0.35, 0.35])  # Larger pie chart 

 

# Loading the dataset for the pie chart 
data = pd.read_csv("preprocessed_customer_shopping_data.csv") 
data['invoice_date'] = pd.to_datetime(data['invoice_date']) 
filtered_data = data[(data['invoice_date'].dt.year >= 2021) & (data['invoice_date'].dt.year <= 2023)] 
yearly_sales = filtered_data.groupby(filtered_data['invoice_date'].dt.year)['price'].sum() 

 

# Compute the total sum of yearly sales
total_sales = yearly_sales.sum()

# Plotting the data as a pie chart
ax_pie.pie(yearly_sales.values, labels=yearly_sales.index, autopct=lambda p: f'${int(p/100.*total_sales):.0f}', startangle=140)

ax_pie.set_title('Total Sales Distribution\nfrom 2021 to 2023', fontsize=12) 

 
plt.show()

#Figure 14

# Create age groups

df['age_group'] = pd.cut(df['age'], bins=[18, 25, 35, 45, 55, df['age'].max()], labels=['18-24', '25-34', '35-44', '45-54', '55+'])
 
# Calculate the total spending for each age group

age_group_spending = df.groupby('age_group')['price'].sum().reset_index()
 
# Calculate the percentage of total spending for each age group

total_spending = age_group_spending['price'].sum()

age_group_spending['percentage'] = (age_group_spending['price'] / total_spending) * 100
# Create the donut chart

plt.figure(figsize=(12, 8))

labels = age_group_spending['age_group']

sizes = age_group_spending['percentage']
 
# Create the pie chart

plt.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=90, colors=plt.cm.tab10.colors)

plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
 
# Draw a white circle at the center to create the donut chart

centre_circle = plt.Circle((0, 0), 0.70, fc='white')

fig = plt.gcf()

fig.gca().add_artist(centre_circle)
 
plt.title('Total Spending by Age Group')

plt.show()
 



#Figure 19
import pandas as pd 

 

# Loading the CSV file into a DataFrame 
df = pd.read_csv("preprocessed_customer_shopping_data.csv") 

 

# Grouping the data by shopping mall and calculating the total revenue for each mall 
mall_revenue = df.groupby('shopping_mall')['price'].sum().reset_index() 

 

# Converting the DataFrame to a list of dictionaries for Highcharts 
data = mall_revenue.to_dict(orient='records') 

 

# Defining the Highcharts options 

options = { 

    'chart': { 

        'type': 'column' 

    }, 

    'title': { 

        'text': 'Total Revenue by Shopping Mall in Turkish liras (₺)' 

    }, 

    'xAxis': { 

        'categories': [mall['shopping_mall'] for mall in data] 

    }, 

    'yAxis': { 

        'title': { 

            'text': 'Total Revenue' 

        } 

    }, 

    'series': [{ 

        'name': 'Revenue', 

        'data': [mall['price'] for mall in data] 

    }] 

} 

 

# Generating the Highcharts HTML content 

highchart_html = f''' 

<!DOCTYPE html> 

<html lang="en"> 

<head> 

  <meta charset="UTF-8"> 

  <title>Highcharts Example</title> 

  <script src="https://code.highcharts.com/highcharts.js"></script> 

</head> 

<body> 

  <div id="container" style="width: 800px; height: 400px;"></div> 

  <script> 

    document.addEventListener('DOMContentLoaded', function () {{ 

      Highcharts.chart('container', {options}); 

    }}); 

  </script> 

</body> 

</html> 

''' 

 

# Writing the HTML content to a file 

with open('highcharts_revenue.html', 'w') as f: 

    f.write(highchart_html) 

print("Highcharts visualization created successfully. Please check the 'highcharts_revenue.html' file.") 


#Figure 21 - Pyspark outputs
# Importing PySpark 
from pyspark.sql import SparkSession 
from pyspark.sql.functions import sum, count, col, when

# Creating a SparkSession 
spark = SparkSession.builder \
    .appName("Customer Shopping Data Analysis") \
    .getOrCreate()

# Loading the dataset into a Spark DataFrame 
df = spark.read.csv('preprocessed_customer_shopping_data.csv', header=True, inferSchema=True)

# Exploring the data 
print("Number of records:", df.count()) 
print("Column names:", df.columns) 

# Grouping data by shopping mall, calculate total spend and number of customers for each mall, and display top 10 malls by total spend 
mall_stats = df.groupBy("shopping_mall") \
               .agg(sum("price").alias("total_spend"), count("customer_id").alias("num_customers")) \
               .orderBy(col("total_spend").desc()) \
               .limit(10) 
print("\nTop 10 malls by total spend:") 
mall_stats.show() 

# Analyzing the data by gender, calculate total spend and number of customers for each gender 
gender_stats = df.groupBy("gender") \
                 .agg(sum("price").alias("total_spend"), count("customer_id").alias("num_customers")) 
print("\nGender analysis:") 
gender_stats.show() 

# Analyzing data by age, calculate total spend and number of customers for each age group 
age_groups = { 
    "18-24": (df.age >= 18) & (df.age <= 24), 
    "25-34": (df.age >= 25) & (df.age <= 34), 
    "35-44": (df.age >= 35) & (df.age <= 44), 
    "45-54": (df.age >= 45) & (df.age <= 54), 
    "55+": df.age >= 55 
}

age_stats = df.select(*[when(condition, age_group).alias(f"age_group_{age_group}") for age_group, condition in age_groups.items()], 
                     col("price"), col("customer_id")) \
              .groupBy(*[f"age_group_{age_group}" for age_group in age_groups.keys()]) \
              .agg(sum("price").alias("total_spend"), count("customer_id").alias("num_customers")) \
              .orderBy(*[f"age_group_{age_group}" for age_group in age_groups.keys()]) 
print("\nAge group analysis:") 
age_stats.show() 
spark.stop()
