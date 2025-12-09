import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


# Page configuration
st.set_page_config(
    page_title="MRT-3 Ridership Dashboard",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the MRT-3 ridership data"""
    # Load the data using absolute path for deployment
    import os
    base_path = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base_path, 'data', 'cleaned_ridership_data.csv'))
    
    # Convert to long format
    df_melted = df.melt(id_vars=['Date', 'Year'], 
                        var_name='Month', 
                        value_name='Ridership')
    
    # Map month names to numbers
    month_to_num = {month: i for i, month in enumerate(calendar.month_name[1:], 1)}
    df_melted['Month'] = df_melted['Month'].map(month_to_num)
    
    # Replace 0 with NaN for better analysis
    df_melted = df_melted.replace(0, np.nan)
    
    # Create datetime column for better time series analysis (use actual day)
    df_melted['Date_Obj'] = pd.to_datetime(
        df_melted[['Year', 'Month', 'Date']].rename(columns={'Year': 'year', 'Month': 'month', 'Date': 'day'}),
        errors='coerce'
    )
    
    return df_melted

def main():
    # Load data
    df_melted = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">🚇 MRT-3 Ridership Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Comprehensive Analysis of Metro Rail Transit Line 3 Ridership Data (1999-March 2025)</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Year range selector
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df_melted['Year'].min()),
        max_value=int(df_melted['Year'].max()),
        value=(int(df_melted['Year'].min()), int(df_melted['Year'].max())),
        step=1
    )
    
    # Filter data based on year range
    filtered_df = df_melted[(df_melted['Year'] >= year_range[0]) & (df_melted['Year'] <= year_range[1])]
    
    # Analysis type selector
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Overview", "Trend Analysis", "Seasonal Patterns", "Statistical Analysis", "Predictions"]
    )
    
    # Main content based on analysis type
    if analysis_type == "Overview":
        show_overview(filtered_df, df_melted)
    elif analysis_type == "Trend Analysis":
        show_trend_analysis(filtered_df)
    elif analysis_type == "Seasonal Patterns":
        show_seasonal_patterns(filtered_df)
    elif analysis_type == "Statistical Analysis":
        show_statistical_analysis(filtered_df)
    elif analysis_type == "Predictions":
        show_predictions(filtered_df)

def show_overview(filtered_df, full_df):
    """Display overview dashboard with key metrics and charts"""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_ridership = int(filtered_df['Ridership'].sum())
        st.metric("Total Ridership", f"{total_ridership:,}")
    
    with col2:
        avg_daily = int(filtered_df['Ridership'].mean())
        st.metric("Average Daily Ridership", f"{avg_daily:,}")
    
    with col3:
        max_daily = int(filtered_df['Ridership'].max())
        st.metric("Peak Daily Ridership", f"{max_daily:,}")
    
    with col4:
        total_days = len(filtered_df.dropna())
        st.metric("Days with Data", f"{total_days:,}")
    
    st.markdown("---")
    
    # Yearly trends
    st.markdown('<h2 class="section-header">📈 Annual Ridership Trends</h2>', unsafe_allow_html=True)
    
    yearly_data = filtered_df.groupby('Year')['Ridership'].agg(['sum', 'mean', 'max']).reset_index()
    yearly_data.columns = ['Year', 'Total_Ridership', 'Avg_Daily', 'Peak_Daily']
    # Convert to integers where appropriate
    yearly_data['Total_Ridership'] = yearly_data['Total_Ridership'].astype(int)
    yearly_data['Peak_Daily'] = yearly_data['Peak_Daily'].astype(int)
    yearly_data['Avg_Daily'] = yearly_data['Avg_Daily'].astype(int)
    
    # Calculate year-over-year growth
    yearly_data['YoY_Growth'] = yearly_data['Total_Ridership'].pct_change() * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(yearly_data, x='Year', y='Total_Ridership', 
                     title='Annual Total Ridership',
                     labels={'Total_Ridership': 'Total Ridership', 'Year': 'Year'})
        fig.update_layout(height=400)
        # Format y-axis to show integers
        fig.update_yaxes(tickformat=",")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Filter out extreme growth values for better visualization
        growth_data = yearly_data.copy()
        # Cap extreme values at ±200% for better visualization
        growth_data['YoY_Growth_Capped'] = growth_data['YoY_Growth'].clip(-200, 200)
        
        fig = px.bar(growth_data, x='Year', y='YoY_Growth_Capped',
                    title='Year-over-Year Growth (%) - Capped at ±200%',
                    labels={'YoY_Growth_Capped': 'Growth (%)', 'Year': 'Year'},
                    color='YoY_Growth_Capped',
                    color_continuous_scale='RdYlGn')
        
        # Add annotations for extreme values
        for idx, row in growth_data.iterrows():
            if abs(row['YoY_Growth']) > 200:
                fig.add_annotation(
                    x=row['Year'], y=200 if row['YoY_Growth'] > 0 else -200,
                    text=f"{row['YoY_Growth']:.0f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="red"
                )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly patterns
    st.markdown('<h2 class="section-header">📅 Monthly Ridership Patterns</h2>', unsafe_allow_html=True)
    
    monthly_data = filtered_df.groupby(['Year', 'Month'])['Ridership'].sum().reset_index()
    monthly_data['Month_Name'] = monthly_data['Month'].map(lambda x: calendar.month_name[x])
    
    fig = px.imshow(
        monthly_data.pivot(index='Year', columns='Month_Name', values='Ridership'),
        title='Monthly Ridership Heatmap',
        labels={'x': 'Month', 'y': 'Year', 'color': 'Ridership'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown('<h2 class="section-header">🔍 Key Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📊 Data Coverage:**")
        st.markdown(f"- Total years: {len(yearly_data)}")
        st.markdown(f"- Date range: {yearly_data['Year'].min()} - {yearly_data['Year'].max()}")
        st.markdown(f"- Peak year: {yearly_data.loc[yearly_data['Total_Ridership'].idxmax(), 'Year']}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**📈 Growth Trends:**")
        best_growth = yearly_data.loc[yearly_data['YoY_Growth'].idxmax()]
        worst_growth = yearly_data.loc[yearly_data['YoY_Growth'].idxmin()]
        # Check if 2025 is the best growth year and add disclaimer
        if best_growth['Year'] == 2025:
            st.markdown(f"- Best growth: {best_growth['Year']} ({best_growth['YoY_Growth']:.0f}%) - **Incomplete Data**")
        else:
            st.markdown(f"- Best growth: {best_growth['Year']} ({best_growth['YoY_Growth']:.0f}%)")
        # Check if 2025 is the worst decline year and add disclaimer
        if worst_growth['Year'] == 2025:
            st.markdown(f"- Worst decline: {worst_growth['Year']} ({worst_growth['YoY_Growth']:.0f}%) - **Incomplete Data**")
        else:
            st.markdown(f"- Worst decline: {worst_growth['Year']} ({worst_growth['YoY_Growth']:.0f}%)")
        st.markdown("</div>", unsafe_allow_html=True)

def show_trend_analysis(filtered_df):
    """Display trend analysis with various time series visualizations"""
    
    st.markdown('<h2 class="section-header">📈 Trend Analysis</h2>', unsafe_allow_html=True)
    
    # Time series plot
    st.markdown('<h3>Daily Ridership Time Series</h3>', unsafe_allow_html=True)
    
    # Create a proper time series
    ts_data = filtered_df.copy()
    ts_data['Date_Obj'] = pd.to_datetime(
        ts_data[['Year', 'Month', 'Date']].rename(columns={'Year': 'year', 'Month': 'month', 'Date': 'day'}),
        errors='coerce'
    )
    ts_data = ts_data.sort_values('Date_Obj')
    
    fig = px.line(ts_data, x='Date_Obj', y='Ridership',
                  title='Daily Ridership Over Time',
                  labels={'Date_Obj': 'Date', 'Ridership': 'Daily Ridership'})
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Daily Ridership")
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    # Moving averages
    st.markdown('<h3>Moving Averages</h3>', unsafe_allow_html=True)
    
    # Calculate moving averages
    ts_data_sorted = ts_data.sort_values('Date_Obj').reset_index(drop=True)
    ts_data_sorted['MA_30'] = ts_data_sorted['Ridership'].rolling(window=30).mean()
    ts_data_sorted['MA_90'] = ts_data_sorted['Ridership'].rolling(window=90).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_data_sorted['Date_Obj'], y=ts_data_sorted['Ridership'],
                            mode='lines', name='Daily Ridership', opacity=0.6))
    fig.add_trace(go.Scatter(x=ts_data_sorted['Date_Obj'], y=ts_data_sorted['MA_30'],
                            mode='lines', name='30-Day Moving Average', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=ts_data_sorted['Date_Obj'], y=ts_data_sorted['MA_90'],
                            mode='lines', name='90-Day Moving Average', line=dict(color='green')))
    
    fig.update_layout(title='Ridership with Moving Averages',
                     height=500, xaxis_title="Date", yaxis_title="Ridership")
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    # Decade comparison
    st.markdown('<h3>Decade Comparison</h3>', unsafe_allow_html=True)
    
    filtered_df['Decade'] = (filtered_df['Year'] // 10) * 10
    decade_data = filtered_df.groupby('Decade')['Ridership'].agg(['mean', 'std', 'count']).reset_index()
    decade_data.columns = ['Decade', 'Mean_Ridership', 'Std_Ridership', 'Data_Points']
    decade_data['Mean_Ridership'] = decade_data['Mean_Ridership'].astype(int)
    decade_data['Std_Ridership'] = decade_data['Std_Ridership'].astype(int)
    decade_data['Data_Points'] = decade_data['Data_Points'].astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
            fig = px.bar(decade_data, x='Decade', y='Mean_Ridership',
                title='Average Daily Ridership by Decade',
                labels={'Mean_Ridership': 'Average Daily Ridership', 'Decade': 'Decade'})
    fig.update_layout(height=400)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
            fig = px.scatter(decade_data, x='Decade', y='Mean_Ridership', size='Data_Points',
                    title='Decade Analysis with Data Points',
                    labels={'Mean_Ridership': 'Average Daily Ridership', 'Decade': 'Decade', 'Data_Points': 'Data Points'})
    fig.update_layout(height=400)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

def show_seasonal_patterns(filtered_df):
    """Display seasonal and monthly patterns"""
    
    st.markdown('<h2 class="section-header">📅 Seasonal Patterns Analysis</h2>', unsafe_allow_html=True)
    
    # Monthly patterns
    st.markdown('<h3>Monthly Ridership Patterns</h3>', unsafe_allow_html=True)
    
    monthly_avg = filtered_df.groupby('Month')['Ridership'].mean().reset_index()
    monthly_avg['Month_Name'] = monthly_avg['Month'].map(lambda x: calendar.month_name[x])
    monthly_avg['Ridership'] = monthly_avg['Ridership'].astype(int)
    
    fig = px.bar(monthly_avg, x='Month_Name', y='Ridership',
                title='Average Daily Ridership by Month',
                labels={'Ridership': 'Average Daily Ridership', 'Month_Name': 'Month'})
    fig.update_layout(height=500)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly heatmap by year
    st.markdown('<h3>Monthly Ridership Heatmap by Year</h3>', unsafe_allow_html=True)
    
    monthly_by_year = filtered_df.groupby(['Year', 'Month'])['Ridership'].mean().reset_index()
    monthly_by_year['Month_Name'] = monthly_by_year['Month'].map(lambda x: calendar.month_name[x])
    
    heatmap_data = monthly_by_year.pivot(index='Year', columns='Month_Name', values='Ridership')
    
    fig = px.imshow(heatmap_data,
                    title='Monthly Average Ridership Heatmap',
                    labels={'x': 'Month', 'y': 'Year', 'color': 'Average Daily Ridership'},
                    aspect='auto')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Day of month patterns
    st.markdown('<h3>Day of Month Patterns</h3>', unsafe_allow_html=True)
    
    day_patterns = filtered_df.groupby('Date')['Ridership'].mean().reset_index()
    day_patterns['Ridership'] = day_patterns['Ridership'].astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
            fig = px.line(day_patterns, x='Date', y='Ridership',
                 title='Average Ridership by Day of Month',
                 labels={'Ridership': 'Average Daily Ridership', 'Date': 'Day of Month'})
    fig.update_layout(height=400)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
            fig = px.scatter(day_patterns, x='Date', y='Ridership',
                    title='Day of Month Scatter Plot',
                    labels={'Ridership': 'Average Daily Ridership', 'Date': 'Day of Month'})
    fig.update_layout(height=400)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal insights
    st.markdown('<h3>Seasonal Insights</h3>', unsafe_allow_html=True)
    
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    filtered_df['Season'] = filtered_df['Month'].apply(get_season)
    seasonal_data = filtered_df.groupby('Season')['Ridership'].agg(['mean', 'std', 'count']).reset_index()
    seasonal_data['mean'] = seasonal_data['mean'].astype(int)
    seasonal_data['std'] = seasonal_data['std'].astype(int)
    seasonal_data['count'] = seasonal_data['count'].astype(int)
    
    fig = px.bar(seasonal_data, x='Season', y='mean',
                title='Average Daily Ridership by Season',
                labels={'mean': 'Average Daily Ridership', 'Season': 'Season'})
    fig.update_layout(height=400)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

def show_statistical_analysis(filtered_df):
    """Display statistical analysis and distributions"""
    
    st.markdown('<h2 class="section-header">📊 Statistical Analysis</h2>', unsafe_allow_html=True)
    
    # Basic statistics
    st.markdown('<h3>Descriptive Statistics</h3>', unsafe_allow_html=True)
    
    stats = filtered_df['Ridership'].describe()
    # Convert to integers where appropriate, round floats to 2 decimal places
    stats_formatted = pd.Series({
        'count': int(stats['count']),
        'mean': int(stats['mean']),
        'std': int(stats['std']),
        'min': int(stats['min']),
        '25%': int(stats['25%']),
        '50%': int(stats['50%']),
        '75%': int(stats['75%']),
        'max': int(stats['max'])
    })
    st.dataframe(stats_formatted)
    
    # Distribution plots
    st.markdown('<h3>Ridership Distribution</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
            fig = px.histogram(filtered_df, x='Ridership', nbins=50,
                      title='Ridership Distribution Histogram',
                      labels={'Ridership': 'Daily Ridership', 'count': 'Frequency'})
    fig.update_layout(height=400)
    # Format x-axis to show integers
    fig.update_xaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    with col2:
            fig = px.box(filtered_df, y='Ridership',
                title='Ridership Box Plot',
                labels={'Ridership': 'Daily Ridership'})
    fig.update_layout(height=400)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.markdown('<h3>Correlation Analysis</h3>', unsafe_allow_html=True)
    
    corr_data = filtered_df[['Year', 'Month', 'Date', 'Ridership']].corr()
    
    fig = px.imshow(corr_data,
                    title='Correlation Matrix',
                    labels={'x': 'Variables', 'y': 'Variables', 'color': 'Correlation'},
                    aspect='auto')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Outlier analysis
    st.markdown('<h3>Outlier Analysis</h3>', unsafe_allow_html=True)
    
    Q1 = filtered_df['Ridership'].quantile(0.25)
    Q3 = filtered_df['Ridership'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = filtered_df[(filtered_df['Ridership'] < lower_bound) | (filtered_df['Ridership'] > upper_bound)]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Outlier Statistics:**")
        st.markdown(f"- Lower bound: {int(lower_bound):,}")
        st.markdown(f"- Upper bound: {int(upper_bound):,}")
        st.markdown(f"- Number of outliers: {len(outliers)}")
        st.markdown(f"- Outlier percentage: {(len(outliers)/len(filtered_df)*100):.1f}%")
    
    with col2:
        if len(outliers) > 0:
            fig = px.scatter(outliers, x='Date_Obj', y='Ridership',
                           title='Outliers in Ridership Data',
                           labels={'Ridership': 'Daily Ridership', 'Date_Obj': 'Date'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def show_predictions(filtered_df):
    """Display predictive analysis and forecasting"""
    
    st.markdown('<h2 class="section-header">🔮 Predictive Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**Note:** This section provides trend analysis using polynomial regression (degree 3) to capture non-linear patterns in the data. Projections are based on historical trends and should be interpreted with caution.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Simple trend projection
    st.markdown('<h3>Trend Projection</h3>', unsafe_allow_html=True)
    
    yearly_totals = filtered_df.groupby('Year')['Ridership'].sum().reset_index()
    
    # Use polynomial regression for more realistic trend (degree 3 for flexibility)
    z = np.polyfit(yearly_totals['Year'], yearly_totals['Ridership'], 3)
    p = np.poly1d(z)
    
    # Project next 5 years
    future_years = np.arange(yearly_totals['Year'].max() + 1, yearly_totals['Year'].max() + 6)
    future_predictions = p(future_years)
    
    # Create projection plot
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(x=yearly_totals['Year'], y=yearly_totals['Ridership'],
                            mode='lines+markers', name='Historical Data',
                            line=dict(color='blue')))
    
    # Trend line (curved)
    fig.add_trace(go.Scatter(x=yearly_totals['Year'], y=p(yearly_totals['Year']),
                            mode='lines', name='Trend Line (Polynomial)',
                            line=dict(color='red', dash='dash')))
    
    # Future projections (curved)
    fig.add_trace(go.Scatter(x=future_years, y=future_predictions,
                            mode='lines+markers', name='Projections (Polynomial)',
                            line=dict(color='green', dash='dot')))
    
    fig.update_layout(title='Ridership Trend and Projections',
                     height=500, xaxis_title="Year", yaxis_title="Total Annual Ridership")
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)
    
    # Projection table
    st.markdown('<h3>5-Year Projections</h3>', unsafe_allow_html=True)
    
    projection_df = pd.DataFrame({
        'Year': future_years,
        'Projected_Ridership': future_predictions.astype(int),
        'Growth_Rate': np.round(np.diff(np.concatenate([yearly_totals['Ridership'].iloc[-1:], future_predictions])) / yearly_totals['Ridership'].iloc[-1] * 100, 0).astype(int)
    })
    
    st.dataframe(projection_df)
    
    # Seasonal forecasting
    st.markdown('<h3>Seasonal Forecasting</h3>', unsafe_allow_html=True)
    
    # Monthly averages for forecasting
    monthly_forecast = filtered_df.groupby('Month')['Ridership'].mean().reset_index()
    monthly_forecast['Month_Name'] = monthly_forecast['Month'].map(lambda x: calendar.month_name[x])
    monthly_forecast['Ridership'] = monthly_forecast['Ridership'].astype(int)
    
    fig = px.bar(monthly_forecast, x='Month_Name', y='Ridership',
                title='Monthly Average Ridership (Basis for Seasonal Forecasting)',
                labels={'Ridership': 'Average Daily Ridership', 'Month_Name': 'Month'})
    fig.update_layout(height=400)
    # Format y-axis to show integers
    fig.update_yaxes(tickformat=",")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
