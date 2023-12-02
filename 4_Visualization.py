import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV, train_test_split
from surprise import SVD, accuracy
from collections import defaultdict

pip install matplotlib

# Load datasets
dataset_clustering = pd.read_csv('./Dataset Streamlit/dataset_clustering.csv', sep=';')
raw_data_merge = pd.read_csv('./Dataset Streamlit/raw_data_merge.csv', sep=',')
monthly_purchase = pd.read_csv('./Dataset Streamlit/monthly_purchase.csv', sep=',')
hourly_purchase_count = pd.read_csv('./Dataset Streamlit/hourly_purchase_count.csv', sep=',')
data_rfm_fix = pd.read_csv('./Dataset Streamlit/data_rfm_fix.csv', sep=',')
top_categories_df = pd.read_csv('./Dataset Streamlit/top_categories_df.csv', sep=',')
df = pd.read_csv('./Dataset Streamlit/df.csv', sep=',')
merged_df = pd.read_csv('./Dataset Streamlit/merged_df.csv', sep=',')
top_10_ratings = pd.read_csv('./Dataset Streamlit/top_10_ratings.csv', sep=',')

# Set page config
st.set_page_config(
    page_title="Hello",
)

# Menu navigation
page = st.radio("Select a page", ["Welcome", "Sample Raw Dataset", "Clean Dataset", "Key Metrics Overview", "Top Categories", "Clusters Overview", "Top Categories by Cluster", "Recommendation System"])

# Page 1 - Welcome
if page == "Welcome":
    st.markdown("# Welcome to Portfolio Julius Kevin! 👋")
    st.markdown("## Improve e-commerce Sustainabilities through Clustering and Recommendation System")
    st.markdown("### By Julius Kevin")
    st.markdown("###Data Scientist with a BSc in Industrial Engineering. Proven expertise in data analysis, project management, and Google Ads. Runner-up in Data Science Boothcamp. Proficient in Python, SQL, and Looker Studio. Seeking a role to apply skills and drive innovation.")

# Page 2 - Sample Raw Dataset
elif page == "Sample Raw Dataset":
    st.title('Sample Raw Dataset')
    st.write(raw_data_merge)

    # Add explanation
    total_entries = raw_data_merge.shape[0]
    st.write(f"Total entries in the dataset: {total_entries}")

    # Features to drop with less than 10% missing values
    features_to_drop_10_percent = ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date',
                                   'product_weight_g', 'product_length_cm', 'product_width_cm']
    st.write("Features with less than 10% missing values will be dropped:")
    st.write(features_to_drop_10_percent)

    # Features to drop with >50% missing values
    features_to_drop_50_percent = ['review_comment_title', 'review_comment_message']
    st.write("Features with more than 50% missing values will be dropped:")
    st.write(features_to_drop_50_percent)

# Page 3 - Clean Dataset
elif page == "Clean Dataset":
    st.title('Clean Dataset')
    st.write(dataset_clustering)

    # Add explanation
    total_entries_clean_data = dataset_clustering.shape[0]
    st.write(f"Total entries in the clean dataset: {total_entries_clean_data}")
    st.write("No missing values")
    st.write("No duplicate values")

# Page 4 - Key Metrics Overview
elif page == "Key Metrics Overview":
    metrics = ['frequency', 'monetary', 'avg_score', 'total_review', 'total_product_order', 'total_category_order', 'recency']

    def display_metrics_overview(data, metrics):
        overview = pd.DataFrame(columns=['Metric', 'Mean', 'Min', 'Max', 'Median'])
        
        for metric in metrics:
            mean_value = data[metric].mean()
            min_value = data[metric].min()
            max_value = data[metric].max()
            median_value = data[metric].median()
            
            overview = overview.append({'Metric': metric, 'Mean': mean_value, 'Min': min_value, 'Max': max_value, 'Median': median_value}, ignore_index=True)
        
        return overview

    st.title('Key Metrics Overview')
    overview_table = display_metrics_overview(dataset_clustering, metrics)
    st.table(overview_table)
    st.markdown('One customer only purchases one time. This data explained that most customers still rarely purchased')
    st.markdown('Same with total review, product order, and category order which means e-commerce still not yet optimized based on customer needs')

    st.write(monthly_purchase)

# Page 5 - Top Categories
elif page == "Top Categories":
    st.title('Top 10 Product Categories with Highest Number of Sales Amount')
    category_sales_amount = monthly_purchase.groupby('product_category_name_english')['payment_value'].sum()
    top_10_categories_salesamount = category_sales_amount.nlargest(10).reset_index()

    chart = alt.Chart(top_10_categories_salesamount).mark_bar().encode(
        x=alt.X('product_category_name_english:N', title='Product Category', sort='-y'),
        y=alt.Y('payment_value:Q', title='Purchase Amount'),
        tooltip=['product_category_name_english:N', 'payment_value:Q']
    ).properties(
        width=600,
        height=400
    )
    st.altair_chart(chart, use_container_width=True)

    st.title('Total Sales and Trend per Month & Category')

    def purchase_amount_by_category(data, category):
        if category == 'All':
            grouped_data = data.groupby(['order_year', 'order_month'])['payment_value'].sum().reset_index()
            grouped_data['hari'] = 1
            grouped_data['year_month'] = grouped_data.apply(lambda row: f"{int(row['order_year'])}-{int(row['order_month']):02d}-{int(row['hari']):02d}", axis=1)
    
            fig, ax = plt.subplots(figsize=(10, 6))
            st.line_chart(grouped_data, y='payment_value', x='year_month')
            st.markdown('**All**\n'
                        'The trend is increasing starting from 2016 with the peak of sales was on Nov’17 due to Black Friday')
        else:
            df_category = data[data['product_category_name_english'] == category]
            grouped_data = df_category.groupby(['order_year', 'order_month', 'product_category_name_english'])['payment_value'].sum().reset_index()
        
            grouped_data['hari'] = 1
            grouped_data['year_month'] = grouped_data.apply(lambda row: f"{int(row['order_year'])}-{int(row['order_month']):02d}-{int(row['hari']):02d}", axis=1)
    
            fig, ax = plt.subplots(figsize=(10, 6))
            st.line_chart(grouped_data, y='payment_value', x='year_month')
            
            if category == 'air_conditioning':
                st.markdown('**Air Conditioning**\n'
                            'Key takeaways:\n'
                            '• Sales have been trending upwards since 2016, with a slight dip in mid-2018.\n'
                            '• The air conditioning category has been the top-selling category throughout the period shown.\n'
                            '• Sales tend to be higher in the summer months (November to March).\n'
                            '• Sales tend to be lower in the winter months (April to October).\n'
                            'Recommendations:\n'
                            '• Focus marketing and sales efforts on the air conditioning category, as it is the most popular category.\n'
                            '• Target marketing campaigns to the summer months, when sales are typically higher.\n'
                            '• Consider offering discounts or promotions during the winter months to boost sales.\n'
                            'Additional insights:\n'
                            '• The graph shows a seasonal pattern in sales, with sales being higher in the summer months and lower in the winter months. This is likely due to the fact that Brazil has a tropical climate, and air conditioners are in higher demand during the hotter months.')
            elif category == 'audio':
                st.markdown('**Audio**\n'
                            'Key takeaways:\n'
                            '• Total sales of audio products in Brazil have been trending downwards since 2016.\n'
                            '• The biggest drop in sales occurred in 2018, with sales decreasing by over 20% compared to the previous year.\n'
                            'Recommendations:\n'
                            '• Businesses in the audio industry in Brazil need to find ways to reverse the downward trend in sales. This could involve developing new and innovative products or targeting new customer segments.\n'
                            '• Businesses should also consider offering discounts or promotions to boost sales.')
            # Add explanations for other categories...
            elif category == 'baby':
                st.markdown('**Baby**\n'
                            'Key takeaways:\n'
                            '• Sales have been trending upwards since 2016.\n'
                            '• Sales have been increasing steadily over time.\n'
                            '• There is no seasonal pattern in sales.\n'
                            'Recommendations:\n'
                            '• Develop new and innovative products to keep up with the growing demand.\n'
                            'Additional insights:\n'
                            '• There is no clear seasonality in sales of baby products, which could be due to the fact that babies need baby products all year round.')
            elif category == 'bed_bath_table':
                st.markdown('**Bed Bath Table**\n'
                            'Key takeaways:\n'
                            '• Total sales of bed bath tables in Brazil have been trending slightly upwards since 2016, with a few dips in sales along the way.\n'
                            '• Sales tend to be higher in the summer months and lower in the winter months.\n'
                            '• The biggest dip in sales occurred in mid-2018.\n'
                            'Recommendations:\n'
                            '• Focus marketing and sales efforts on the summer months, when sales are typically higher.\n'
                            '• Consider offering discounts or promotions during the winter months to boost sales.\n'
                            '• Develop new and innovative bed bath table products to appeal to consumers in the Brazilian market.\n'
                            'Additional insights:\n'
                            '• The dip in sales of bed bath tables in mid-2018 may be due to a number of factors, such as the economic recession that Brazil was experiencing at the time, or the increase in the number of competitors in the bed bath table market.\n'
                            '• The seasonal pattern in sales is consistent with the fact that Brazil has a tropical climate.')
            elif category == 'computers_accessories':
                st.markdown('**Computers and Accessories**\n'
                            'Key takeaways:\n'
                            '• Total sales of computers and accessories in Brazil have been trending upwards since 2016.\n'
                            '• There was a slight dip in sales in mid-2018.\n'
                            'Recommendations:\n'
                            '• Businesses should also develop new and innovative computer and accessory products to appeal to consumers in the Brazilian market.')
            elif category == 'console_games':
                st.markdown('**Console Games**\n'
                            'Key takeaways:\n'
                            '• Total sales of games in Brazil have been trending upwards since 2016.\n'
                            '• The biggest increase in sales occurred in 2021, with sales increasing by over 30% compared to the previous year.\n'
                            'Recommendations:\n'
                            '• Businesses should also develop new and innovative mobile games that integrated with console games to appeal to different customer segments.\n'
                            '• Businesses should also focus on marketing and sales efforts for the mobile games category since it is the fastest growing category around the world.\n'
                            'Additional insights:\n'
                            '• The increase in sales of games in Brazil in recent years is likely due to a number of factors, such as the improvement in the Brazilian economy and the increasing availability of smartphones and tablets.\n'
                            '• The decline in sales of console games is likely due to the increasing popularity of mobile games, which are seen as being more convenient and affordable.')
            elif category == 'cool_stuff':
                st.markdown('**Cool Stuff**\n'
                            'Key takeaways:\n'
                            '• Total sales of cool stuff in Brazil have been trending upwards since 2016, with a slight dip in mid-2018.\n'
                            '• There were slight dips in sales in 2018 but overall trend was positive.\n'
                            'Recommendations:\n'
                            '• Businesses should also develop new and innovative cool stuff products to appeal to consumers in the Brazilian market.')
            elif category == 'diapers_and_hygiene':
                st.markdown('**Diapers and Hygiene**\n'
                            'Key takeaways:\n'
                            '• Total sales of diapers and hygiene in Brazil have been trending upwards since 2016.\n'
                            '• Sales of diapers and hygiene tend to be higher in the summer months and lower in the winter months.\n'
                            'Recommendations:\n'
                            '• Businesses should also develop new and innovative diapers and hygiene products to appeal to consumers in the Brazilian market.\n'
                            '• Businesses should also focus on marketing and sales efforts for the diapers and hygiene category, especially during the summer months.')
            elif category == 'electronics':
                st.markdown('**Electronics**\n'
                            'Key takeaways:\n'
                            '• Total sales of electronics in Brazil have been trending upwards since 2016.\n'
                            '• There were slight dips in sales in 2018 but overall trend was positive.\n'
                            'Recommendations:\n'
                            '• Businesses should also develop new and innovative electronics products to appeal to consumers in the Brazilian market.\n'
                            '• Businesses should also focus on marketing and sales efforts for the electronics category.\n'
                            'Additional insights:\n'
                            '• The rise in popularity of smartphones around the world could be a signal of factors, such as the increasing affordability of smartphones, the growing penetration of mobile data, and the increasing use of smartphones for entertainment and productivity.')
            elif category == 'fashio_female_clothing':
                st.markdown('**Fashion Female Clothing**\n'
                            'Key takeaways:\n'
                            '• Total sales of fashion jewelry in Brazil have been trending upwards since 2016.\n'
                            '• Sales of fashion jewelry tend to be higher in the summer months and lower in the winter months.\n'
                            '• There was a slight dip in sales in 2018 but overall trend was positive.\n'
                            'Recommendations:\n'
                            '• Businesses should also develop new and innovative fashion jewelry products to appeal to consumers in the Brazilian market.\n'
                            '• Businesses should also focus on marketing and sales efforts for the fashion jewelry category, especially during the summer months.\n'
                            'Additional insights:\n'
                            '• The seasonal pattern in sales of fashion jewelry is likely due to the fact that Brazil has a tropical climate and people tend to wear more fashion jewelry in the summer months to express their personal style.')
            elif category == 'fashion_bags_accessories':
                st.markdown('**Fashion Bags and Accessories**\n'
                            'Key takeaways:\n'
                            '• Total sales of fashion bags and accessories in Brazil have been trending upwards since 2016.\n'
                            '• Sales tend to be higher in the summer months and lower in the winter months.\n'
                            'Recommendations:\n'
                            '• Businesses should also develop new and innovative fashion bags and accessories products to appeal to consumers in the Brazilian market.\n'
                            '• Businesses should also focus on marketing and sales efforts for the fashion bags and accessories category, especially during the summer months.\n'
                            'Additional insights:\n'
                            '• The seasonal pattern in sales of fashion bags and accessories is likely due to the fact that Brazil has a tropical climate and people tend to carry more bags in the summer months to carry essentials such as sunscreen and water bottle.')

    categories_list = ['All'] + list(monthly_purchase['product_category_name_english'].unique())
    selected_category = st.selectbox('Select a category', categories_list)
    purchase_amount_by_category(monthly_purchase, selected_category)

    st.title('Hourly Purchase Count by Category')

    hourly_purchase_count = pd.read_csv('./Dataset Streamlit/hourly_purchase_count.csv', sep=',')
    st.write(hourly_purchase_count)

    def hourly_purchase_count_by_category(data, category):
        df_category = data[data['product_category_name_english'] == category]
        grouped_data = df_category.groupby(['order_hour', 'product_category_name_english'])['order_id'].sum().reset_index()

        # Create a figure and adjust figure size
        fig, ax = plt.subplots(figsize=(10, 6))
        
        st.line_chart(grouped_data.sort_values('order_hour'), x='order_hour', y='order_id')

    # Call the function with the desired category column
    selected_category_hourly = st.selectbox('Select a category', hourly_purchase_count['product_category_name_english'].unique())
    hourly_purchase_count_by_category(hourly_purchase_count, selected_category_hourly)

# Page 6 - Clusters Overview
elif page == "Clusters Overview":
    st.title('3 Clusters based on RFM')
    st.write(data_rfm_fix)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
    sns.boxplot(x='clusters', y='frequency', data=data_rfm_fix, ax=axes[0])
    axes[0].set_title('Frequency Distribution by Cluster')
    sns.boxplot(x='clusters', y='recency', data=data_rfm_fix, ax=axes[1])
    axes[1].set_title('Recency Distribution by Cluster')
    sns.boxplot(x='clusters', y='monetary', data=data_rfm_fix, ax=axes[2])
    axes[2].set_title('Monetary Distribution by Cluster')
    plt.tight_layout()
    st.pyplot(fig)

    # Rename clusters
    data_rfm_fix['cluster_labels'] = data_rfm_fix['clusters'].map({
        0: 'Almost Churn Customer',
        1: 'Top Customer',
        2: 'New Customer'
    })

    # 3D Scatter Plot
    scatter_3d_fig = px.scatter_3d(
        data_rfm_fix,
        x='frequency',
        y='monetary',
        z='recency',
        color='cluster_labels',
        size_max=10,
        opacity=0.7,
        title='3D Scatter Plot: Frequency vs Monetary vs Recency by Clusters',
        labels={'frequency': 'Frequency', 'monetary': 'Monetary', 'recency': 'Recency'}
    )

    # Show the plot
    st.plotly_chart(scatter_3d_fig)

    # Add cluster explanations
    st.markdown('**Cluster Explanations:**')
    st.markdown('- **Cluster 0:** Not engaged customers. (1x Transactions, Already registered a long time ago)')
    st.markdown('- **Cluster 1:** Top customers (Loyal customers)')
    st.markdown('- **Cluster 2:** New customers (Warm customers)')

# Page 7 - Top Categories by Cluster
elif page == "Top Categories by Cluster":
    st.title('Top 5 categories based on each cluster')
    st.write(top_categories_df)
    
    cluster_mapping = {0: 'Almost Churn Customer', 1: 'Top Customer', 2: 'New Customer'}
    top_categories_df['clusters'] = top_categories_df['clusters'].map(cluster_mapping)

    colors = sns.color_palette("viridis", len(top_categories_df))

    selected_cluster = st.selectbox('Select Cluster', top_categories_df['clusters'].unique())
    selected_cluster_data = top_categories_df[top_categories_df['clusters'] == selected_cluster]

    # Sort the data by frequency in descending order
    selected_cluster_data = selected_cluster_data.sort_values(by='order_id', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(selected_cluster_data['product_category_name_english'], selected_cluster_data['order_id'], color=colors)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Product Category')
    ax.set_title(f'Top 5 Product Categories for Cluster: {selected_cluster}')
    st.pyplot(fig)

    # Explanation
    st.header('Cluster Analysis and Recommendations')
    
    common_top_categories = ['Bed_bath_table', 'Health_beauty', 'Sports_leisure']
    unique_top_categories = {
        'Almost Churn': ['Computer_accessories', 'Furniture_decor'],
        'Top Customers': ['Furniture_decor', 'Computer_accessories'],
        'New Customers': ['Watches_gifts', 'Housewares']
    }

    st.markdown('**Common Top Categories for Each Cluster:**')
    st.write(common_top_categories)

    st.markdown('**Unique Top Categories for Each Cluster:**')
    for cluster, categories in unique_top_categories.items():
        st.write(f'Cluster {cluster}: {categories}')

    st.markdown('**Insights and Recommendations:**')
    st.write("1. Almost churn and Top customers have similar categories, indicating potential similarities in expectations. "
             "Consider exploring commonalities in product approach, buying experience, and marketing messages.")

    st.write("2. Recommendation: Develop promotion strategies tailored to each cluster and product category. "
             "Collaborate with sellers in different categories for bundling packages.")

# Page 8 - Recommendation System
elif page == "Recommendation System":
    st.title('Recommendation System')

    # Rename clusters
    top_10_ratings['Cluster'] = top_10_ratings['Cluster'].map({
        0: 'Almost Churn Customer',
        1: 'Top Customer',
        2: 'New Customer'
    })

    cluster_selection = st.selectbox('Select Cluster', top_10_ratings['Cluster'].unique())

    # Display selected cluster's data in a table
    selected_cluster_data = top_10_ratings[top_10_ratings['Cluster'] == cluster_selection]
    st.table(selected_cluster_data)

    
