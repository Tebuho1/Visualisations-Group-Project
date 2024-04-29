import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import folium
from geopy.geocoders import Nominatim
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE



#For Figure 17 (Visualisation 15) - the running time is about 5-7 minutes due to the large dataset
#We tried and ran all of the following codes on Jupyter

# Load the data
df = pd.read_csv('preprocessed_customer_shopping_data.csv')

# Group the data by shopping mall and category
mall_category_counts = df.groupby(['shopping_mall', 'category'])['invoice_no'].count().unstack(fill_value=0)

# Create a stacked bar chart
plt.figure(figsize=(16, 10))  # Set the figure size
mall_category_counts.plot(kind='bar', stacked=True)  # Plot a stacked bar chart
plt.title('Transactions by Shopping Mall and Category')  # Add title to the plot
plt.xlabel('Shopping Mall')  # Add label for the x-axis
plt.ylabel('Number of Transactions')  # Add label for the y-axis
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Add legend with specified location
plt.show()  # Display the plot
print("This is visualization 5: figure 7")  # Print information about the visualization

# Create a pivot table to get the transaction counts by age group and shopping mall
pivot_table = df.pivot_table(index=pd.cut(df['age'], bins=range(0, 81, 10)), columns='shopping_mall', values='invoice_no', aggfunc='count')

# Create a heatmap to visualize the transaction counts
plt.figure(figsize=(12, 8))  # Set the figure size
sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='g')  # Plot the heatmap with annotations
plt.title('Transactions by Age Group and Shopping Mall')  # Add title to the plot
plt.xlabel('Shopping Mall')  # Add label for the x-axis
plt.ylabel('Age Group')  # Add label for the y-axis
plt.show()  # Display the plot
print("This is visualization 6: figure 8")  # Print information about the visualization

import pandas as pd
import folium
from geopy.geocoders import Nominatim
import webbrowser
 
# Load the data
df = pd.read_csv('preprocessed_customer_shopping_data.csv')
 
# Filter the data to only include shopping malls in Istanbul
istanbul_malls = ['Mall of Istanbul', 'Istinye Park', 'Kanyon', 'Metrocity', 'Emaar Square Mall', 'Metropol AVM', 'Zorlu Center', 'Forum Istanbul', 'Cevahir AVM', 'Viaport Outlet']
df = df[df['shopping_mall'].isin(istanbul_malls)]
 
# Get the gender distribution for each shopping mall
mall_gender_distribution = df.groupby('shopping_mall')['gender'].value_counts(normalize=True).unstack(fill_value=0)
 
# Create a dictionary to store the coordinates
mall_coordinates = {}
 
# Initialize the Nominatim geocoder
geolocator = Nominatim(user_agent="my_app")
 
# Geocode each shopping mall and store the coordinates
for mall in istanbul_malls:
    try:
        location = geolocator.geocode(f"{mall}, Istanbul")
        if location:
            mall_coordinates[mall] = (location.latitude, location.longitude)
    except:
        print(f"Error geocoding {mall}")
 
# Calculate the average latitude and longitude of the malls to center the map
avg_lat = sum([coord[0] for coord in mall_coordinates.values()]) / len(mall_coordinates)
avg_lon = sum([coord[1] for coord in mall_coordinates.values()]) / len(mall_coordinates)
map_center = (avg_lat, avg_lon)
 
# Create the folium map object
map_obj = folium.Map(location=map_center, zoom_start=12)
 
# Define the color ranges and their corresponding labels for the legend
color_ranges = {
    '0 - 6000': 'red',
    '6000 - 12000': 'orange',
    '12000 - 18000': 'green',
    '18000+': 'blue'
}
 
# Create the legend HTML
legend_html = '''
<div style="position: fixed; 
                 top: 50px; right: 50px; width: 200px; height: 120px; 
                 border:2px solid grey; z-index:9999; font-size:14px;
                 background-color:white;
                 ">
&nbsp; <strong>Customer Count</strong> <br>
     '''
 
# Add color ranges to the legend
for label, color in color_ranges.items():
    legend_html += f"&nbsp; {label}: <i style='background:{color}'>&nbsp;&nbsp;&nbsp;</i><br>"
 
legend_html += '''
</div>
     '''
 
# Add the legend HTML to the map
map_obj.get_root().html.add_child(folium.Element(legend_html))
 
# Add markers to the map with color-coded density and gender distribution in the popup
for mall, coord in mall_coordinates.items():
    # Get gender distribution percentages for the current mall
    male_percentage = mall_gender_distribution.loc[mall, 'Male'] * 100
    female_percentage = mall_gender_distribution.loc[mall, 'Female'] * 100
    # Count the number of customers for the current mall
    mall_count = df[df['shopping_mall'] == mall].shape[0]
 
    # Determine the color for the marker based on the customer count
    if mall_count < 6000:
        color = 'red'
    elif mall_count < 12000:
        color = 'orange'
    elif mall_count < 18000:
        color = 'green'
    else:
        color = 'blue'
 
    # Create popup text for the marker
    popup_text = f"""
    Shopping Mall: {mall}
    Customer Count: {mall_count}
    Male Percentage: {male_percentage:.2f}%
    Female Percentage: {female_percentage:.2f}%
    """
 
    # Add marker to the map with the determined color and popup text
    folium.Marker(
        coord,
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color)
    ).add_to(map_obj)
 

 # Save the map as an HTML file
map_obj.save('customer_shopping_map.html')
 
# Open the HTML file in a new tab
webbrowser.open('customer_shopping_map.html')

 
print("Customershopping visualization created successfully. Please check the 'customer_shopping_map.html' file.") 
 
print("This is visualization 7: figure 9")
 
# Create a pivot table to get the transaction counts by age group and payment method
pivot_table = df.pivot_table(index=pd.cut(df['age'], bins=range(0, 81, 10)), columns='payment_method', values='invoice_no', aggfunc='count')

# Create a heatmap to visualize the transaction counts
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='g')
plt.title('Transactions by Payment Method and Age Group')
plt.xlabel('Payment Method')
plt.ylabel('Age Group')
plt.show()

# Print information about the visualization
print("This is visualization 9: figure 11")


# Group the data by gender and shopping mall, and calculate the total spending
gender_mall_spending = df.groupby(['gender', 'shopping_mall'])['price'].sum().reset_index()

# Create the group bar chart
plt.figure(figsize=(18, 10))
ax = sns.barplot(
    x="shopping_mall",  # X-axis: Shopping Mall
    y="price",  # Y-axis: Total Spending
    hue="gender",  # Group by Gender
    data=gender_mall_spending,  # Data Source
    palette="viridis"  # Color Palette
)

# Add the numeric values of the bars with some space
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 5.0, f'{height:,.2f}', ha="center")

# Chart Formatting
plt.title("Total Spending by Gender and Shopping Mall")
plt.xlabel("Shopping Mall")
plt.ylabel("Total Spending ")
plt.xticks(rotation=90)
plt.legend(title="Gender")
plt.tight_layout()

# Display the plot
plt.show()

# Print information about the visualization
print("This is visualization 10: figure 12")


# Select the desired categories
desired_categories = ['Clothing', 'Food & Beverage', 'Cosmetics', 'Shoes', 'Technology']

# Create the faceted histograms
fig, axes = plt.subplots(len(desired_categories), len(df['gender'].unique()), figsize=(16, 12))
for i, category in enumerate(desired_categories):
    for j, gender in enumerate(df['gender'].unique()):
        # Filter the data for the current category and gender
        filtered_df = df[(df['category'] == category) & (df['gender'] == gender)]
        # Create a histogram with kernel density estimation (kde)
        sns.histplot(x='age', data=filtered_df, kde=True, ax=axes[i, j], color=sns.color_palette('Set2')[j])
        # Set title for each subplot
        axes[i, j].set_title(f"{category} - {gender}")

# Set the title for the entire visualization
plt.suptitle('Spending Distribution by Age, Category, and Gender')
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()  # Display the plot

# Print information about the visualization
print("This is visualization 11: figure 13")

# Create a pivot table to get the transaction counts by age group and gender
pivot_table = df.pivot_table(index=pd.cut(df['age'], bins=range(0, 81, 10)), columns='gender', values='invoice_no', aggfunc='count')

# Create a heatmap to visualize the transaction counts
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='g')
plt.title('Transactions by Age Group and Gender')
plt.xlabel('Gender')
plt.ylabel('Age Group')
plt.show()

# Print information about the visualization
print("This is visualization 13: figure 15")


# Select the relevant features for clustering
features = ['age', 'price']
X = df[features]

# Apply k-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create the scatter plot
plt.figure(figsize=(12, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.title('Clustering Analysis of Customers')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster')
plt.show()

# Print information about the visualization
print("This is visualization 15: figure 17")
