# ko_kolobeng_demo.py (Updated to use Store Sales Time Series Forecasting dataset)

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import holidays # For South African holidays
import kagglehub
from kagglehub import KaggleDatasetAdapter
import warnings

# Suppress warnings that might appear during Streamlit or Prophet operations
warnings.filterwarnings("ignore", category=DeprecationWarning, module="kagglehub")
warnings.filterwarnings("ignore", category=FutureWarning, module="prophet")
warnings.filterwarnings("ignore", category=UserWarning, module="prophet")


# --- 1. Data Loading and Preprocessing (Cached for performance) ---
@st.cache_data # Cache the data loading and initial processing
def load_and_preprocess_data():
    """
    Loads the alternative 'Store Sales - Time Series Forecasting' dataset from Kaggle Hub
    and performs initial preprocessing.
    Returns a DataFrame with daily product sales, or an empty DataFrame if loading fails.
    """
    df = None # Initialize df to None outside the try block

    try:
        # --- IMPORTANT CHANGE: Load the NEW ALTERNATIVE dataset ---
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "store-sales-time-series-forecasting", # Competition dataset identifier
            "train.csv" # The primary training data file
        )
    except Exception as e:
        st.error(f"Error loading dataset: {e}. Please check your internet connection or KaggleHub setup.")
        return pd.DataFrame() # Return an empty DataFrame to prevent UnboundLocalError

    if df is None or df.empty:
        st.warning("No data loaded. Please ensure the dataset exists and can be accessed.")
        return pd.DataFrame()

    # --- IMPORTANT CHANGE: Adapt column names and aggregation for new dataset ---
    # Original columns: 'date', 'store_nbr', 'family', 'sales', 'onpromotion'
    df['date'] = pd.to_datetime(df['date'])
    df['invoice_date'] = df['date'].dt.date # Extract date only for daily aggregation

    # Group by date and 'family' (which represents product categories)
    # Sum 'sales' as the quantity for a given 'family' on a given day
    daily_product_sales = df.groupby(['invoice_date', 'family'])['sales'].sum().reset_index()
    daily_product_sales['invoice_date'] = pd.to_datetime(daily_product_sales['invoice_date']) # Convert back to datetime for Prophet

    # Rename columns to match expected for Prophet
    daily_product_sales = daily_product_sales.rename(columns={'family': 'category', 'sales': 'quantity'})

    return daily_product_sales

# --- 2. Prophet Model Training (Cached for performance) ---
@st.cache_resource(ttl="1h") # Cache the trained model for 1 hour
def train_prophet_model(df_sales, selected_product_name):
    """
    Trains a Prophet model for a given product name.
    Returns the trained model and its forecast, or (None, None) if data is insufficient.
    """
    product_df = df_sales[df_sales['category'] == selected_product_name].copy()

    if product_df.empty:
        return None, None # No data for this product

    start_date = df_sales['invoice_date'].min()
    end_date = df_sales['invoice_date'].max()
    full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    full_product_df = pd.DataFrame({'invoice_date': full_date_range})
    full_product_df['category'] = selected_product_name

    product_df_filled = pd.merge(full_product_df, product_df, on=['invoice_date', 'category'], how='left')
    product_df_filled['quantity'] = product_df_filled['quantity'].fillna(0)

    prophet_df = product_df_filled.rename(columns={'invoice_date': 'ds', 'quantity': 'y'})

    if prophet_df['y'].count() < 2 or prophet_df[prophet_df['y'] > 0].shape[0] < 2:
        return None, None

    holiday_start_year = prophet_df['ds'].min().year
    holiday_end_year = prophet_df['ds'].max().year + 2
    sa_holidays = holidays.country_holidays(
        'ZA',
        years=range(holiday_start_year, holiday_end_year)
    )
    sa_holidays_df = pd.DataFrame({
        'holiday': 'ZA_Holiday',
        'ds': pd.to_datetime(list(sa_holidays.keys())),
        'lower_window': 0,
        'upper_window': 0,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=sa_holidays_df
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)

    return model, forecast

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Ko Kolobeng: Smart Restaurant Insights Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Ko Kolobeng: Smart Restaurant Insights Demo 🇿🇦")
st.markdown("### Powered by Data Science & Machine Learning for Optimal Operations")

# Sidebar for navigation
st.sidebar.header("Navigation")
page_selection = st.sidebar.radio(
    "Choose a section:",
    ["📈 Dish Demand Forecasting", "🍽️ Ingredient Inventory Management", "🤝 Bulk Buying Advantage"]
)

daily_product_sales_df = load_and_preprocess_data()

# --- Map actual item names from the new dataset to demo-friendly names ---
if not daily_product_sales_df.empty:
    actual_item_names_in_data = sorted(daily_product_sales_df['category'].unique().tolist())
else:
    actual_item_names_in_data = []

# Mapping from dataset 'family' names to restaurant-friendly display names
# This list is based on common categories in the 'Store Sales' dataset
item_name_mapping = {
    'GROCERY I': 'Staple Groceries (e.g., Pap Flour)',
    'BEVERAGES': 'Beverages (e.g., Soft Drinks)',
    'PRODUCE': 'Fresh Produce (e.g., Vegetables)',
    'MEATS': 'Meat Products (e.g., Beef/Chicken)',
    'PREPARED FOODS': 'Prepared Dishes (e.g., Stews)',
    'DAIRY': 'Dairy Products',
    'FROZEN FOODS': 'Frozen Items',
    'BAKERY': 'Bakery Items',
    'DELI': 'Deli & Prepared Foods',
    'BREAD/BAKERY': 'Bread & Baked Goods'
}

demo_products_for_selection = []
for item_name in actual_item_names_in_data:
    if item_name in item_name_mapping:
        demo_products_for_selection.append(item_name_mapping[item_name])
    # Optionally, include unmapped categories if you want a wider selection, but might be less restaurant-specific
    # else:
    #     demo_products_for_selection.append(item_name)

# Fallback if no mapped items are found (shouldn't happen with this dataset but good practice)
if not demo_products_for_selection and actual_item_names_in_data:
    demo_products_for_selection = actual_item_names_in_data[:min(10, len(actual_item_names_in_data))]
elif not actual_item_names_in_data:
    demo_products_for_selection = ["(No items found in dataset)"]

# Create a reverse mapping for internal use
reverse_item_mapping = {v: k for k, v in item_name_mapping.items()}


# --- Page 1: Dish Demand Forecasting ---
if page_selection == "📈 Dish Demand Forecasting":
    st.header("Predicting What Dishes & Ingredients Your Customers Will Need")
    st.write(
        "Never run out of popular items or have too much unused inventory. "
        "This tool predicts how much of each product category you'll need in the coming weeks. "
        "It helps identify typical sales patterns and potential spikes, useful for event planning."
    )

    st.markdown("---")

    if not daily_product_sales_df.empty:
        selected_display_product = st.selectbox(
            "Select a Product Category to Forecast:",
            demo_products_for_selection,
            index=0 if demo_products_for_selection and "(No items found in dataset)" not in demo_products_for_selection else 0
        )

        if selected_display_product and selected_display_product != "(No items found in dataset)":
            selected_internal_product = reverse_item_mapping.get(selected_display_product, selected_display_product)

            with st.spinner(f"Generating forecast for {selected_display_product}..."):
                model, forecast_df = train_prophet_model(daily_product_sales_df, selected_internal_product)

                if model and forecast_df is not None:
                    st.subheader(f"Sales Forecast for '{selected_display_product}'")

                    fig = px.line(
                        forecast_df,
                        x='ds',
                        y='yhat',
                        title=f'Predicted Daily Sales for {selected_display_product}',
                        labels={'ds': 'Date', 'yhat': 'Predicted Quantity Sold'}
                    )
                    fig.add_scatter(x=model.history['ds'], y=model.history['y'], mode='markers', name='Actual Sales',
                                     marker=dict(size=4, opacity=0.6))
                    fig.update_layout(hovermode="x unified", legend_title_text="Legend")
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("#### Forecasted Quantities for the Next 30 Days:")
                    last_actual_sale_date = daily_product_sales_df['invoice_date'].max()
                    future_forecast = forecast_df[forecast_df['ds'] > last_actual_sale_date].copy()
                    future_forecast['ds'] = future_forecast['ds'].dt.date
                    st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
                                 .rename(columns={'ds': 'Date', 'yhat': 'Predicted Sales',
                                                   'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'})
                                 .set_index('Date')
                    )
                    st.info(
                        "The `Predicted Sales` is the most likely quantity. "
                        "The `Lower Bound` and `Upper Bound` give you a range of possible sales."
                        "\n\n*This demo uses a large, general retail dataset. With your actual restaurant sales data, the forecasts will be even more precise and directly actionable for your specific dishes and ingredients.*"
                    )
                else:
                    st.warning(f"Forecast not available for '{selected_display_product}'. This demo uses a generic dataset, and this item might not have enough consistent sales history within it to generate a reliable forecast. Please try another item.")
        else:
            st.info("Please select a product category to view its demand forecast.")
    else:
        st.error("Cannot display demand forecasting. No data was loaded. Please check data source and internet connection.")


# --- Page 2: Ingredient Inventory Management (Simulated) ---
elif page_selection == "🍽️ Ingredient Inventory Management":
    st.header("Smart Ingredient Inventory Management for Your Restaurant")
    st.write(
        "Stop guessing! Our system helps you visualize your stock levels and provides smart reorder suggestions "
        "to ensure you always have the right ingredients, when you need them, minimizing waste and ensuring customer satisfaction."
    )

    st.markdown("---")

    st.subheader("Simulated Stock Status & Reorder Alerts")
    st.info(
        "This section shows how the system would work with your actual sales data. "
        "We can give you precise, real-time alerts for your key ingredients, minimizing spoilage and maximizing freshness."
    )

    simulated_products_data = {
        'Item': ['Beef (kg)', 'Mielie-Meal (kg)', 'Fresh Veggies (kg)', 'Soft Drinks (case)', 'Chicken (kg)', 'Cooking Oil (L)', 'Spices (kg)', 'Bottled Water (case)', 'Dessert Prep (units)', 'Coffee Beans (kg)'],
        'Current Stock (Units)': [30, 25, 40, 10, 50, 15, 5, 8, 20, 7],
        'Daily Predicted Usage': [15, 10, 20, 5, 20, 3, 1, 4, 8, 2],
        'Reorder Threshold (Days of Usage)': [2, 3, 1, 4, 2, 5, 7, 3, 2, 5]
    }
    simulated_df = pd.DataFrame(simulated_products_data)

    simulated_df['Reorder Point (Units)'] = simulated_df['Daily Predicted Usage'] * simulated_df['Reorder Threshold (Days of Usage)']
    simulated_df['Status'] = '🟢 Good Stock'
    simulated_df['Reorder Suggestion'] = 'None'

    for index, row in simulated_df.iterrows():
        if row['Current Stock (Units)'] < row['Reorder Point (Units)']:
            if row['Current Stock (Units)'] <= row['Daily Predicted Usage'] * 1.5:
                simulated_df.loc[index, 'Status'] = '🔴 Critical Stock - Reorder ASAP!'
                simulated_df.loc[index, 'Reorder Suggestion'] = f"Order {row['Daily Predicted Usage'] * 7} units (approx. 1 week supply)"
            else:
                simulated_df.loc[index, 'Status'] = '🟡 Low Stock - Consider Reordering'
                simulated_df.loc[index, 'Reorder Suggestion'] = f"Order {row['Daily Predicted Usage'] * 5} units (approx. 5 days supply)"

    st.dataframe(simulated_df[['Item', 'Current Stock (Units)', 'Daily Predicted Usage', 'Status', 'Reorder Suggestion']]
                 .set_index('Item'))

    st.markdown("""
    **How this helps your restaurant:**
    * **Reduce Food Waste:** Don't buy ingredients that will spoil before you use them.
    * **Ensure Dish Availability:** Always have popular ingredients on hand to avoid disappointing customers.
    * **Optimize Cash Flow:** Don't tie up capital in excessive ingredient inventory.
    * **Efficient Sourcing:** Plan your ingredient purchases better with reliable usage forecasts.
    """)

# --- Page 3: Bulk Buying Advantage (Conceptual) ---
elif page_selection == "🤝 Bulk Buying Advantage":
    st.header("Unlock Bigger Savings with Collective Buying Power")
    st.write(
        "Imagine uniting with other local restaurants. By combining your predicted ingredient demands, "
        "you can place larger orders with suppliers and unlock significant discounts, just like major restaurant chains!"
    )

    st.markdown("---")

    st.subheader("The Power of Aggregation (Conceptual Example)")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Scenario A: Individual Restaurant Ordering")
        st.image("https://via.placeholder.com/300x150?text=KoKolobeng+Order+-+Expensive", caption="Ko Kolobeng Orders Individually")
        st.image("https://via.placeholder.com/300x150?text=Another+Restaurant+Order+-+Expensive", caption="Another Restaurant Orders Individually")
        st.markdown("""
        * Each restaurant orders smaller quantities of ingredients.
        * **Supplier Price:** R150 per kg of Beef.
        * **Result:** Higher unit cost, less profit per dish.
        """)

    with col2:
        st.markdown("#### Scenario B: Collective Ordering (Future Vision)")
        st.image("https://via.placeholder.com/300x150?text=Combined+Restaurant+Order+-+Cheaper", caption="Multiple Restaurants Order Together")
        st.markdown("""
        * Our system aggregates predicted ingredient demand from multiple restaurants.
        * **Combined Order Volume:** Enough to qualify for bulk discount from suppliers.
        * **Supplier Price:** R120 per kg of Beef (example discount).
        * **Result:** **Significant savings for *each* participating restaurant!**
        """)

    st.markdown("---")
    st.markdown("""
    **How this can boost your restaurant's profits:**
    * **Lower Procurement Costs:** Buy quality ingredients at wholesale prices usually reserved for large buyers.
    * **Increased Margins:** Sell dishes at competitive prices while still making more profit.
    * **Reduced Transport Costs:** Potentially optimized delivery for bulk orders.

    This feature would thrive as more restaurants join a collective buying network facilitated by the platform.
    """)
