import streamlit as st
import pandas as pd
import plotly.express as px
#import google.generativeai as genai
st.set_page_config(page_title="Walmart Sales Dashboard", layout="wide")
# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\jay\Downloads\stepclassdata\Walmart.csv")
    return df

df = load_data()

# Sidebar for navigation
st.sidebar.title("📊 Dashboard Navigation")
menu = st.sidebar.radio("Select Analysis", [
    "Category Contribution Analysis (Pie Chart)",
    "Sales Trend Analysis (Line Chart)",
    "Performance Benchmarking (Bar Chart)",
    "Customer Purchase Distribution (Histogram)",
    "📈 Insight Board"
])

# MBA Visualization 1: Category Contribution (Pie Chart)
if menu == "Category Contribution Analysis (Pie Chart)":
    st.subheader("🧩 Category Contribution Analysis")
    if "Category" in df.columns and "Weekly_Sales" in df.columns:
        pie_data = df.groupby("Category")["Weekly_Sales"].sum().reset_index()
        fig = px.pie(pie_data, names='Category', values='Weekly_Sales', title="Sales Contribution by Category")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columns 'Category' and/or 'Weekly_Sales' not found.")

# MBA Visualization 2: Sales Trend (Line Chart)
elif menu == "Sales Trend Analysis (Line Chart)":
    st.subheader("📈 Sales Trend Analysis")
    if "Date" in df.columns and "Weekly_Sales" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        trend_data = df.groupby("Date")["Weekly_Sales"].sum().reset_index()
        fig = px.line(trend_data, x="Date", y="Weekly_Sales", title="Sales Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columns 'Date' and/or 'Weekly_Sales' not found.")

# MBA Visualization 3: Performance Benchmarking (Bar Chart)
elif menu == "Performance Benchmarking (Bar Chart)":
    st.subheader("🏆 Performance Benchmarking by Store")
    if "Store" in df.columns and "Weekly_Sales" in df.columns:
        bar_data = df.groupby("Store")["Weekly_Sales"].sum().reset_index()
        fig = px.bar(bar_data, x='Store', y='Weekly_Sales', title='Total Sales by Store')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Columns 'Store' and/or 'Weekly_Sales' not found.")

# MBA Visualization 4: Customer Purchase Distribution (Histogram)
elif menu == "Customer Purchase Distribution (Histogram)":
    st.subheader("📊 Customer Purchase Distribution")
    if "Weekly_Sales" in df.columns:
        fig = px.histogram(df, x="Weekly_Sales", nbins=30, title="Distribution of Weekly Sales")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Column 'Weekly_Sales' not found.")

# Insight Board (Gemini API)
'''elif menu == "📈 Insight Board":
    st.subheader("📌 Insight Board: AI-Generated Summary")

    with st.spinner("Generating insights using Gemini..."):
        # Setup Gemini
        genai.configure(api_key="YOUR_GEMINI_API_KEY")
        model = genai.GenerativeModel("gemini-pro")

        prompt = f"""You are a business analyst assistant. Analyze the following Walmart sales data and generate a concise general summary with business insights, sales trends, and any notable patterns:\n\n{df.head(100).to_string()}"""

        response = model.generate_content(prompt)
        st.success("Insights Generated:")
        st.write(response.text)

    st.info("This summary is generated using Google's Gemini API. You can extend it for custom queries.")'''

