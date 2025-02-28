import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Price Oracle", layout="wide")

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 10px;
    }
    .subheader {
        font-size: 20px;
        color: #6c757d;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header Section ---
st.markdown('<div class="header">Welcome to Price Oracle</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-Powered Price Predictions and Insights</div>', unsafe_allow_html=True)

# --- Tabs for Navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏆 About Us", "💻 Our Project", "📈 Achievements", "📞 Contact", "📊 Predictions"]
)

# --- About Us Tab ---
with tab1:
    st.header("About Us")
    st.write("Welcome to **Price Oracle**, where **data meets intelligence**! 🚀")
    
    st.write("""
        We are a team of passionate developers, data enthusiasts, and problem-solvers dedicated to building innovative solutions for real-world challenges. 🌍💡 Inspired by the dynamic and ever-changing nature of markets, we created Price Oracle to empower businesses with the tools they need to thrive in uncertainty. 📊📈
    """)

    st.write("### **Our Mission**")
    st.write("""
        - **Predict. Adapt. Succeed.** 🔮  
          With cutting-edge AI technologies, real-time economic insights, and intuitive design, Price Oracle helps businesses make smarter pricing decisions. Whether it's navigating inflation 📉, adapting to GDP shifts 🌐, or forecasting product prices 📦, our platform ensures you're always ahead of the curve.
    """)

    st.write("### **What We Believe In** 🌟")
    st.write("""
        - **Innovation**: Harnessing the power of AI to drive meaningful change. 🤖✨  
        - **Simplicity**: Making complex data accessible and actionable for everyone. 🛠️📂  
        - **Impact**: Helping businesses succeed in volatile markets with confidence. 💼🔥
    """)

    st.write("### **Meet Price Oracle** 💻🔍")
    st.write("""
        - Predict future prices with advanced machine learning models. 🤖📊  
        - Analyze real-time macroeconomic trends like inflation and GDP growth. 🌍💹  
        - Get actionable recommendations to optimize pricing strategies. 🧠✅  
        - Detect anomalies in historical data for better decision-making. ⚠️📉  

        At Price Oracle, we believe that data is power—and we’re here to put that power in your hands. 🖐️💪 Let’s shape the future of pricing together! 🚀
    """)

# --- Our Project Tab ---
with tab2:
    st.header("Our Project")
    st.write("""
        Price Oracle uses advanced machine learning models to analyze historical pricing data, macroeconomic trends, and real-time market dynamics.
        By integrating predictive analytics with explainable insights, we help businesses make informed decisions about pricing optimization.
    """)
    
    # Add a video demo (replace with your project demo video link)
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

# --- Achievements Tab ---
with tab3:
    st.header("Our Achievements")
    st.write("""
        - 🏆 Winner of Hacklytics 2025
        - 🥇 Best AI-Powered Application Award
        - 🎖️ Recognized for Innovation in Predictive Analytics
    """)
    
    # Add charts or visuals for achievements
    chart_data = {"Category": ["Innovation", "Impact", "Technical Depth"], "Score": [95, 90, 88]}
    df = pd.DataFrame(chart_data)
    
    st.bar_chart(df.set_index("Category"))

# --- Contact Tab ---
with tab4:
    st.header("Contact Us")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("📧 Email: team@priceoracle.tech")
        st.write("🌐 Website: www.priceoracle.tech")
    
    with col2:
        st.write("📍 Location: Georgia Tech, Atlanta")
        st.write("📞 Phone: +1 (123) 456-7890")

# --- Predictions Tab ---
with tab5:
    st.header("AI-Powered Predictions")
    
    # Inputs and functionality exclusive to the Predictions Tab
    device = st.selectbox(
        "Choose iPhone 12 Model",
        [
            "iPhone 12 Mini 64GB", "iPhone 12 Mini 128GB", "iPhone 12 Mini 256GB",
            "iPhone 12 64GB", "iPhone 12 128GB", "iPhone 12 256GB",
            "iPhone 12 Pro 128GB", "iPhone 12 Pro 256GB", "iPhone 12 Pro Max"
        ]
    )
    
    months = st.slider("📅 Months to Predict", min_value=1, max_value=6, value=3)
    
    # Simulate Inflation Rate and GDP Growth Inputs
    inflation_rate = np.random.uniform(1.5, 5.0)  # Random inflation rate for testing
    gdp_growth = np.random.uniform(0.5, 3.5)      # Random GDP growth rate for testing

    # Display User Inputs in Columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Inflation Rate (%)", value=f"{inflation_rate:.2f}")
        
    with col2:
        st.metric(label="GDP Growth (%)", value=f"{gdp_growth:.2f}")

    # Add Generate Predictions Button
    if st.button("🚀 Generate Predictions"):
        
        # Simulate Prediction Data Without Backend
        start_date = datetime.today()
        dates = [start_date + timedelta(days=30 * i) for i in range(months)]
        
        # Generate random predicted prices (for testing purposes)
        predicted_prices = np.random.uniform(600, 1200, size=months).round(2)
        
        # Create a DataFrame for visualization
        prediction_data = pd.DataFrame({
            "Month": [date.strftime("%B %Y") for date in dates],
            "Predicted Price (USD)": predicted_prices,
        })
        
        # Display Results
        st.success(f"✅ Predicted prices generated successfully!")
        
        # Plotly Line Chart
        fig = px.line(
            prediction_data,
            x="Month",
            y="Predicted Price (USD)",
            title="📊 Predicted Prices Over Time",
            markers=True,
            line_shape="spline",
            color_discrete_sequence=["#FF5733"]
        )
        
        fig.update_layout(template="plotly_white")
        
        # Display Chart and Table
        st.plotly_chart(fig)
        
        st.write("### Prediction Data")
        st.dataframe(prediction_data)

# --- Footer Section ---
st.markdown('<div class="footer">Hacklytics 2025 | Price Oracle</div>', unsafe_allow_html=True)
