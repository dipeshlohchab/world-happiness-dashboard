import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="World Happiness Index Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data function
@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration"""
    countries = ['Denmark', 'Switzerland', 'Iceland', 'Norway', 'Finland', 'Canada', 'Netherlands', 
                'New Zealand', 'Australia', 'Sweden', 'United States', 'Germany', 'United Kingdom',
                'France', 'Japan', 'South Korea', 'Brazil', 'Mexico', 'India', 'China', 'Russia',
                'South Africa', 'Nigeria', 'Kenya', 'Egypt']
    
    regions = {
        'Denmark': 'Western Europe', 'Switzerland': 'Western Europe', 'Iceland': 'Western Europe',
        'Norway': 'Western Europe', 'Finland': 'Western Europe', 'Canada': 'North America',
        'Netherlands': 'Western Europe', 'New Zealand': 'Australia and New Zealand',
        'Australia': 'Australia and New Zealand', 'Sweden': 'Western Europe',
        'United States': 'North America', 'Germany': 'Western Europe', 'United Kingdom': 'Western Europe',
        'France': 'Western Europe', 'Japan': 'East Asia', 'South Korea': 'East Asia',
        'Brazil': 'Latin America and Caribbean', 'Mexico': 'Latin America and Caribbean',
        'India': 'South Asia', 'China': 'East Asia', 'Russia': 'Central and Eastern Europe',
        'South Africa': 'Sub-Saharan Africa', 'Nigeria': 'Sub-Saharan Africa',
        'Kenya': 'Sub-Saharan Africa', 'Egypt': 'Middle East and Northern Africa'
    }
    
    data = []
    base_ranks = {country: i+1 for i, country in enumerate(countries)}
    
    for year in range(2015, 2025):
        for i, country in enumerate(countries):
            # Generate realistic data with some variation
            base_score = 8.0 - (i * 0.25) + np.random.normal(0, 0.1)
            rank_variation = np.random.randint(-2, 3)
            
            data.append({
                'Country': country,
                'Year': year,
                'Region': regions[country],
                'Happiness Rank': max(1, base_ranks[country] + rank_variation),
                'Happiness Score': max(2.0, min(8.0, base_score)),
                'GDP': max(0.5, min(2.0, base_score * 0.25 + np.random.normal(0, 0.05))),
                'Social Support': max(0.3, min(1.0, base_score * 0.12 + np.random.normal(0, 0.03))),
                'Life Expectancy': max(0.2, min(1.0, base_score * 0.11 + np.random.normal(0, 0.02))),
                'Freedom': max(0.1, min(0.8, base_score * 0.10 + np.random.normal(0, 0.04))),
                'Generosity': max(-0.1, min(0.6, np.random.normal(0.1, 0.05))),
                'Corruption': max(0.0, min(0.6, 0.4 - base_score * 0.05 + np.random.normal(0, 0.03)))
            })
    
    return pd.DataFrame(data)

# Analysis functions
def calculate_correlations(df):
    """Calculate correlations between happiness score and other factors"""
    factors = ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
    correlations = {}
    
    for factor in factors:
        corr, p_value = pearsonr(df['Happiness Score'], df[factor])
        correlations[factor] = {'correlation': corr, 'p_value': p_value}
    
    return correlations

def get_country_trend(df, country):
    """Get happiness trend for a specific country"""
    country_data = df[df['Country'] == country].sort_values('Year')
    return country_data

def get_regional_analysis(df):
    """Analyze happiness by region"""
    regional_stats = df.groupby('Region').agg({
        'Happiness Score': ['mean', 'std', 'min', 'max'],
        'GDP': 'mean',
        'Social Support': 'mean',
        'Life Expectancy': 'mean',
        'Freedom': 'mean',
        'Generosity': 'mean',
        'Corruption': 'mean'
    }).round(3)
    
    return regional_stats

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 World Happiness Index Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = pd.read_excel('Data/WHRFinal.xlsx')  # Load from CSV if available
    except FileNotFoundError:
        st.warning("Sample data not found. Generating sample data for demonstration.")
        # Generate sample data if CSV not found
        # In a real application, you would load actual data here
        df = load_sample_data()
    
    numeric_cols = ['Happiness Rank', 'Happiness Score', 'GDP', 'Social Support', 
                'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']


    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Dashboard Controls")
    
    # Year selection
    years = sorted(df['Year'].unique())
    selected_years = st.sidebar.slider(
        "Select Year Range",
        min_value=min(years),
        max_value=max(years),
        value=(min(years), max(years)),
        step=1
    )
    
    # Filter data by selected years
    filtered_df = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Page",
        ["📊 Overview & Global Trends", "🏆 Country Rankings", "🔍 Country Deep Dive", 
         "⚖️ Country Comparison", "🌎 Regional Analysis", "📈 Factor Analysis", "🔮 Insights & Predictions"]
    )
    
    if page == "📊 Overview & Global Trends":
        overview_page(filtered_df)
    elif page == "🏆 Country Rankings":
        rankings_page(filtered_df)
    elif page == "🔍 Country Deep Dive":
        country_deep_dive(filtered_df)
    elif page == "⚖️ Country Comparison":
        comparison_page(filtered_df)
    elif page == "🌎 Regional Analysis":
        regional_analysis_page(filtered_df)
    elif page == "📈 Factor Analysis":
        factor_analysis_page(filtered_df)
    elif page == "🔮 Insights & Predictions":
        insights_page(filtered_df)

def overview_page(df):
    st.markdown('<h2 class="section-header">Global Happiness Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_happiness = df['Happiness Score'].mean()
        st.metric("Global Average Happiness", f"{avg_happiness:.2f}", "📊")
    
    with col2:
        happiest_country = df.loc[df['Happiness Score'].idxmax(), 'Country']
        st.metric("Happiest Country", happiest_country, "🥇")
    
    with col3:
        total_countries = df['Country'].nunique()
        st.metric("Countries Analyzed", total_countries, "🌍")
    
    with col4:
        years_span = df['Year'].nunique()
        st.metric("Years of Data", years_span, "📅")
    
    # Global trends over time
    st.subheader("🔄 Global Happiness Trends Over Time")
    
    yearly_stats = df.groupby('Year').agg({
        'Happiness Score': ['mean', 'std'],
        'GDP': 'mean',
        'Social Support': 'mean',
        'Life Expectancy': 'mean'
    }).round(3)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_stats.index,
        y=yearly_stats[('Happiness Score', 'mean')],
        mode='lines+markers',
        name='Average Happiness Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Global Average Happiness Score Trend",
        xaxis_title="Year",
        yaxis_title="Happiness Score",
        height=450,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution of happiness scores
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Happiness Score Distribution")
        fig_hist = px.histogram(
            df, x='Happiness Score', nbins=30,
            title="Distribution of Happiness Scores",
            color_discrete_sequence=['#ff7f0e']
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Regional Distribution")
        region_counts = df['Region'].value_counts()
        fig_pie = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            title="Countries by Region"
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

def rankings_page(df):
    st.markdown('<h2 class="section-header">Country Rankings Dashboard</h2>', unsafe_allow_html=True)
    
    # Year selection for rankings
    selected_year = st.selectbox("Select Year for Rankings", sorted(df['Year'].unique(), reverse=True))
    year_df = df[df['Year'] == selected_year].copy()
    year_df = year_df.sort_values('Happiness Score', ascending=False).reset_index(drop=True)
    year_df['Rank'] = range(1, len(year_df) + 1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🏆 Top 20 Happiest Countries ({selected_year})")
        
        # Top 20 countries bar chart
        top_20 = year_df.head(20)
        fig_bar = px.bar(
            top_20, 
            x='Happiness Score', 
            y='Country',
            orientation='h',
            color='Happiness Score',
            color_continuous_scale='RdYlGn',
            title=f"Top 20 Happiest Countries in {selected_year}"
        )
        fig_bar.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        st.subheader("📈 Ranking Changes")
        
        if len(df['Year'].unique()) > 1:
            # Calculate ranking changes
            prev_year = selected_year - 1
            if prev_year in df['Year'].values:
                prev_df = df[df['Year'] == prev_year][['Country', 'Happiness Score']].copy()
                prev_df = prev_df.sort_values('Happiness Score', ascending=False).reset_index(drop=True)
                prev_df['Prev_Rank'] = range(1, len(prev_df) + 1)
                
                merged = year_df.merge(prev_df[['Country', 'Prev_Rank']], on='Country', how='left')
                merged['Rank_Change'] = merged['Prev_Rank'] - merged['Rank']
                merged = merged.dropna()
                
                # Biggest improvers
                st.write("🚀 **Biggest Improvers:**")
                improvers = merged.nlargest(5, 'Rank_Change')[['Country', 'Rank_Change']]
                for _, row in improvers.iterrows():
                    st.write(f"• {row['Country']}: +{int(row['Rank_Change'])} positions")
                
                st.write("📉 **Biggest Decliners:**")
                decliners = merged.nsmallest(5, 'Rank_Change')[['Country', 'Rank_Change']]
                for _, row in decliners.iterrows():
                    st.write(f"• {row['Country']}: {int(row['Rank_Change'])} positions")
    
    # Rankings table
    st.subheader("📋 Complete Rankings Table")
    display_df = year_df[['Rank', 'Country', 'Region', 'Happiness Score', 'GDP', 'Social Support', 
                         'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']].round(3)
    
    st.dataframe(display_df, use_container_width=True, height=400)

def country_deep_dive(df):
    st.markdown('<h2 class="section-header">Country Deep Dive Analysis</h2>', unsafe_allow_html=True)
    
    # Country selection
    selected_country = st.selectbox("Select Country for Analysis", sorted(df['Country'].unique()))
    country_df = df[df['Country'] == selected_country].sort_values('Year')
    
    if len(country_df) == 0:
        st.error("No data available for selected country.")
        return
    
    # Country overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = country_df.iloc[-1]
    
    with col1:
        st.metric("Current Happiness Score", f"{latest_data['Happiness Score']:.2f}")
    
    with col2:
        st.metric("Current Rank", f"#{int(latest_data['Happiness Rank'])}")
    
    with col3:
        st.metric("Region", latest_data['Region'])
    
    with col4:
        if len(country_df) > 1:
            score_change = latest_data['Happiness Score'] - country_df.iloc[0]['Happiness Score']
            st.metric("Score Change", f"{score_change:+.2f}", delta=f"{score_change:+.2f}")
        else:
            st.metric("Score Change", "N/A")
    
    # Happiness trend over time
    st.subheader(f"📈 {selected_country}'s Happiness Trend")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=country_df['Year'], y=country_df['Happiness Score'], 
                  mode='lines+markers', name='Happiness Score',
                  line=dict(color='#1f77b4', width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=country_df['Year'], y=country_df['Happiness Rank'], 
                  mode='lines+markers', name='Happiness Rank',
                  line=dict(color='#ff7f0e', width=2)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Happiness Score", secondary_y=False)
    fig.update_yaxes(title_text="Happiness Rank", secondary_y=True)
    fig.update_layout(height=400, title=f"{selected_country} - Happiness Score and Rank Over Time")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Factor breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Happiness Factors Breakdown")
        factors = ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity']
        factor_values = [latest_data[factor] for factor in factors]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=factor_values + [factor_values[0]],  # Close the polygon
            theta=factors + [factors[0]],
            fill='toself',
            name=selected_country,
            line_color='#1f77b4'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(factor_values) * 1.1])),
            showlegend=True,
            height=400,
            title="Happiness Factors Radar Chart"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.subheader("📊 Factor Values Over Time")
        
        factors_to_plot = st.multiselect(
            "Select factors to visualize:",
            ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption'],
            default=['GDP', 'Social Support', 'Life Expectancy']
        )
        
        if factors_to_plot:
            fig_factors = go.Figure()
            colors = px.colors.qualitative.Set1
            
            for i, factor in enumerate(factors_to_plot):
                fig_factors.add_trace(go.Scatter(
                    x=country_df['Year'],
                    y=country_df[factor],
                    mode='lines+markers',
                    name=factor,
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig_factors.update_layout(
                height=400,
                title=f"{selected_country} - Selected Factors Over Time",
                xaxis_title="Year",
                yaxis_title="Factor Value"
            )
            
            st.plotly_chart(fig_factors, use_container_width=True)

def comparison_page(df):
    st.markdown('<h2 class="section-header">Country Comparison Analysis</h2>', unsafe_allow_html=True)
    
    # Country selection for comparison
    countries_to_compare = st.multiselect(
        "Select Countries to Compare (2-5 countries recommended):",
        sorted(df['Country'].unique()),
        default=['Denmark', 'United States', 'Japan'] if 'Denmark' in df['Country'].values else df['Country'].unique()[:3].tolist()
    )
    
    if len(countries_to_compare) < 2:
        st.warning("Please select at least 2 countries for comparison.")
        return
    
    # Filter data for selected countries
    comparison_df = df[df['Country'].isin(countries_to_compare)]
    
    # Happiness score comparison over time
    st.subheader("📊 Happiness Score Comparison Over Time")
    
    fig_comparison = px.line(
        comparison_df, 
        x='Year', 
        y='Happiness Score', 
        color='Country',
        markers=True,
        title="Happiness Score Trends Comparison"
    )
    fig_comparison.update_layout(height=400)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Latest year comparison
    latest_year = comparison_df['Year'].max()
    latest_comparison = comparison_df[comparison_df['Year'] == latest_year]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏆 Current Rankings ({latest_year})")
        
        fig_bar_comp = px.bar(
            latest_comparison.sort_values('Happiness Score', ascending=True),
            x='Happiness Score',
            y='Country',
            orientation='h',
            color='Happiness Score',
            color_continuous_scale='RdYlGn',
            title=f"Happiness Scores in {latest_year}"
        )
        fig_bar_comp.update_layout(height=400)
        st.plotly_chart(fig_bar_comp, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Factor Comparison")
        
        factor_to_compare = st.selectbox(
            "Select Factor for Detailed Comparison:",
            ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
        )
        
        fig_factor_comp = px.bar(
            latest_comparison.sort_values(factor_to_compare, ascending=True),
            x=factor_to_compare,
            y='Country',
            orientation='h',
            color=factor_to_compare,
            color_continuous_scale='Viridis',
            title=f"{factor_to_compare} Comparison ({latest_year})"
        )
        fig_factor_comp.update_layout(height=400)
        st.plotly_chart(fig_factor_comp, use_container_width=True)
    
    # Comprehensive comparison table
    st.subheader("📋 Comprehensive Comparison Table")
    
    comparison_table = latest_comparison[['Country', 'Region', 'Happiness Score', 'Happiness Rank',
                                       'GDP', 'Social Support', 'Life Expectancy', 'Freedom', 
                                       'Generosity', 'Corruption']].round(3)
    comparison_table = comparison_table.sort_values('Happiness Score', ascending=False)
    
    st.dataframe(comparison_table, use_container_width=True)
    
    # Statistical comparison
    st.subheader("📈 Statistical Summary")
    
    stats_df = comparison_df.groupby('Country').agg({
        'Happiness Score': ['mean', 'std', 'min', 'max'],
        'GDP': 'mean',
        'Social Support': 'mean',
        'Life Expectancy': 'mean',
        'Freedom': 'mean'
    }).round(3)
    
    st.dataframe(stats_df, use_container_width=True)

def regional_analysis_page(df):
    st.markdown('<h2 class="section-header">Regional Happiness Analysis</h2>', unsafe_allow_html=True)
    
    # Regional overview
    regional_stats = df.groupby('Region').agg({
        'Happiness Score': ['mean', 'std', 'count'],
        'GDP': 'mean',
        'Social Support': 'mean',
        'Life Expectancy': 'mean',
        'Freedom': 'mean',
        'Generosity': 'mean',
        'Corruption': 'mean'
    }).round(3)
    
    regional_means = regional_stats[('Happiness Score', 'mean')].sort_values(ascending=False)
    
    # Regional happiness comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Average Happiness by Region")
        
        fig_regional = px.bar(
            x=regional_means.values,
            y=regional_means.index,
            orientation='h',
            color=regional_means.values,
            color_continuous_scale='RdYlGn',
            title="Regional Average Happiness Scores"
        )
        fig_regional.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_regional, use_container_width=True)
    
    with col2:
        st.subheader("📊 Regional Distribution")
        
        fig_box = px.box(
            df, 
            x='Region', 
            y='Happiness Score',
            title="Happiness Score Distribution by Region"
        )
        fig_box.update_xaxes(tickangle=45)
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Regional trends over time
    st.subheader("📈 Regional Trends Over Time")
    
    regional_yearly = df.groupby(['Region', 'Year'])['Happiness Score'].mean().reset_index()
    
    fig_trends = px.line(
        regional_yearly,
        x='Year',
        y='Happiness Score',
        color='Region',
        markers=True,
        title="Regional Happiness Trends Over Time"
    )
    fig_trends.update_layout(height=500)
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Detailed regional statistics
    st.subheader("📋 Detailed Regional Statistics")
    
    # Flatten column names for better display
    display_stats = regional_stats.copy()
    display_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in display_stats.columns]
    
    st.dataframe(display_stats, use_container_width=True)

def factor_analysis_page(df):
    st.markdown('<h2 class="section-header">Happiness Factors Analysis</h2>', unsafe_allow_html=True)
    
    # Correlation analysis
    st.subheader("🔗 Factor Correlations with Happiness")
    
    correlations = calculate_correlations(df)
    
    # Create correlation dataframe for visualization
    corr_df = pd.DataFrame(correlations).T
    corr_df['Factor'] = corr_df.index
    corr_df = corr_df.sort_values('correlation', key=abs, ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation bar chart
        fig_corr = px.bar(
            corr_df,
            x='correlation',
            y='Factor',
            orientation='h',
            color='correlation',
            color_continuous_scale='RdBu_r',
            title="Correlation with Happiness Score"
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Correlation insights
        st.write("**🔍 Key Insights:**")
        
        strongest_positive = corr_df[corr_df['correlation'] > 0].iloc[0]
        strongest_negative = corr_df[corr_df['correlation'] < 0].iloc[-1] if any(corr_df['correlation'] < 0) else None
        
        st.write(f"• **Strongest positive factor:** {strongest_positive['Factor']} (r = {strongest_positive['correlation']:.3f})")
        
        if strongest_negative is not None:
            st.write(f"• **Strongest negative factor:** {strongest_negative['Factor']} (r = {strongest_negative['correlation']:.3f})")
        
        # Display correlation values
        st.write("**📊 Correlation Values:**")
        for _, row in corr_df.iterrows():
            significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
            st.write(f"• {row['Factor']}: {row['correlation']:.3f}{significance}")
        
        st.caption("* p<0.05, ** p<0.01, *** p<0.001")
    
    # Scatter plots for key relationships
    st.subheader("📈 Factor Relationships")
    
    # Select factors for scatter plot
    factor_x = st.selectbox("Select X-axis factor:", 
                           ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption'],
                           index=0)
    
    factor_y = st.selectbox("Select Y-axis factor:", 
                           ['Happiness Score', 'GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption'],
                           index=0)
    
    # Create scatter plot
    fig_scatter = px.scatter(
        df,
        x=factor_x,
        y=factor_y,
        color='Region',
        size='Happiness Score',
        hover_data=['Country', 'Year'],
        title=f"{factor_y} vs {factor_x}",
        trendline="ols"
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Factor importance over time
    st.subheader("⏱️ Factor Importance Over Time")
    
    # Calculate yearly correlations
    yearly_corr = {}
    factors = ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
    
    for year in df['Year'].unique():
        year_df = df[df['Year'] == year]
        yearly_corr[year] = {}
        for factor in factors:
            if len(year_df) > 2:  # Need at least 3 points for correlation
                corr, _ = pearsonr(year_df['Happiness Score'], year_df[factor])
                yearly_corr[year][factor] = corr
    
    # Convert to DataFrame for plotting
    yearly_corr_df = pd.DataFrame(yearly_corr).T
    yearly_corr_df = yearly_corr_df.reset_index()
    yearly_corr_df = yearly_corr_df.melt(id_vars='index', var_name='Factor', value_name='Correlation')
    yearly_corr_df = yearly_corr_df.rename(columns={'index': 'Year'})
    
    fig_yearly_corr = px.line(
        yearly_corr_df,
        x='Year',
        y='Correlation',
        color='Factor',
        markers=True,
        line_shape='spline',  # smoother lines
        title="Factor Correlations with Happiness Over Time"
    )

    fig_yearly_corr.update_layout(
        template='plotly_white',
        height=500,
        title=dict(x=0.5, font=dict(size=20)),
        xaxis_title="Year",
        yaxis_title="Correlation with Happiness",
        legend_title_text='Factor'
    )

    fig_yearly_corr.for_each_trace(
        lambda t: t.update(line=dict(width=4)) if t.name == "Social Support" else t.update(line=dict(width=2, dash='dot'))
    )

    st.plotly_chart(fig_yearly_corr, use_container_width=True)


def insights_page(df):
    st.markdown('<h2 class="section-header">Key Insights & Analysis</h2>', unsafe_allow_html=True)
    
    # Real-world analysis insights
    st.markdown("""
    <div class="insight-box" style="background-color: #2e6ba0; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4;">
    <h3>🧠 Real-World Happiness Factors Analysis</h3>
    
    Based on extensive research and data analysis, several key factors emerge as critical determinants of national happiness:
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate some key statistics
    correlations = calculate_correlations(df)
    avg_happiness_by_region = df.groupby('Region')['Happiness Score'].mean().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Economic Factors")
        st.markdown("""
        **GDP Per Capita Impact:**
        - Strong correlation with happiness, but diminishing returns beyond middle-income levels
        - Countries with GDP > $30,000 show less dramatic happiness increases
        - Economic security matters more than absolute wealth
        
        **Key Insight:** Money buys happiness up to a point, but social factors become more important for well-being beyond basic needs satisfaction.
        """)
        
        st.subheader("🤝 Social Support Systems")
        st.markdown("""
        **Critical Importance:**
        - Often the strongest predictor of happiness
        - Includes family support, friendship networks, and community bonds
        - Nordic countries excel due to strong social safety nets
        
        **Key Insight:** Having someone to count on in times of need is fundamental to human happiness across all cultures.
        """)
    
    with col2:
        st.subheader("🏥 Health & Longevity")
        st.markdown("""
        **Life Expectancy Correlation:**
        - Reflects overall health system quality
        - Strong correlation with happiness (r ≈ 0.7-0.8)
        - Not just about living longer, but living healthier
        
        **Key Insight:** Investment in healthcare infrastructure directly translates to population well-being.
        """)
        
        st.subheader("🗽 Freedom & Governance")
        st.markdown("""
        **Personal Freedom:**
        - Freedom to make life choices strongly correlates with happiness
        - Includes political freedoms, personal autonomy, and civil liberties
        - Low corruption levels essential for trust in institutions
        
        **Key Insight:** Democratic institutions and rule of law create environments where happiness can flourish.
        """)
    
    # Top performers analysis
    st.subheader("🏆 What Makes Countries Happy: Success Stories")
    
    # Get top 10 happiest countries (average across all years)
    top_countries = df.groupby('Country')['Happiness Score'].mean().nlargest(10)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🇩🇰 Nordic Model Success:**")
        nordic_countries = ['Denmark', 'Finland', 'Norway', 'Sweden', 'Iceland']
        available_nordic = [c for c in nordic_countries if c in df['Country'].values]
        
        if available_nordic:
            nordic_data = df[df['Country'].isin(available_nordic)]
            nordic_avg = nordic_data.groupby('Country')[['Happiness Score', 'GDP', 'Social Support', 'Freedom']].mean()
            
            st.write("Common characteristics:")
            st.write("• High social support")
            st.write("• Low corruption")
            st.write("• Strong welfare systems")
            st.write("• High trust in government")
    
    with col2:
        st.markdown("**🌏 East Asian Insights:**")
        asian_countries = ['Japan', 'South Korea', 'Singapore']
        available_asian = [c for c in asian_countries if c in df['Country'].values]
        
        st.write("Unique factors:")
        st.write("• Economic development focus")
        st.write("• Strong family structures")
        st.write("• Educational achievement")
        st.write("• Social cohesion")
    
    with col3:
        st.markdown("**🌎 Regional Patterns:**")
        st.write("**Top Regions by Happiness:**")
        for i, (region, score) in enumerate(avg_happiness_by_region.head(5).items(), 1):
            st.write(f"{i}. {region}: {score:.2f}")
    
    # Challenges and opportunities
    st.subheader("⚠️ Global Happiness Challenges")
    
    challenges_col1, challenges_col2 = st.columns(2)
    
    with challenges_col1:
        st.markdown("""
        **🌍 Developing World Challenges:**
        - Basic needs fulfillment remains primary concern
        - Economic inequality impacts social cohesion
        - Limited access to healthcare and education
        - Political instability affects long-term planning
        
        **📈 Improvement Opportunities:**
        - Targeted poverty reduction programs
        - Healthcare system development
        - Education access expansion
        - Governance reform initiatives
        """)
    
    with challenges_col2:
        st.markdown("""
        **🏭 Developed World Issues:**
        - Mental health crisis despite material prosperity
        - Social isolation in urban environments
        - Work-life balance deterioration
        - Environmental concerns affecting well-being
        
        **🔄 Potential Solutions:**
        - Community building initiatives
        - Mental health support systems
        - Sustainable development policies
        - Work culture reforms
        """)
    
    # Predictive insights
    st.subheader("🔮 Predictive Factors for Future Happiness")
    
    prediction_insights = """
    **🎯 Key Predictors for National Happiness:**
    
    1. **Social Cohesion Metrics** - Trust levels, community engagement, social capital
    2. **Healthcare Accessibility** - Universal healthcare systems, mental health support
    3. **Educational Quality** - Not just access, but quality and relevance of education
    4. **Environmental Quality** - Clean air, water, sustainable practices
    5. **Economic Equality** - Gini coefficient, income distribution fairness
    6. **Digital Connectivity** - Internet access, digital literacy (increasingly important)
    7. **Cultural Preservation** - Maintaining cultural identity while embracing change
    
    **📊 Emerging Trends:**
    - Climate change impact on happiness becoming more significant
    - Digital well-being and social media effects
    - Remote work implications for community bonds
    - Aging population challenges in developed countries
    """
    
    st.markdown(prediction_insights)
    
    # Policy recommendations
    st.subheader("📋 Evidence-Based Policy Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🏛️ Government Policies:**
        - Invest in mental health infrastructure
        - Implement progressive taxation for equality
        - Strengthen social safety nets
        - Promote work-life balance legislation
        - Support community building initiatives
        """)
    
    with col2:
        st.markdown("""
        **🌱 Long-term Strategies:**
        - Sustainable development integration
        - Education system modernization
        - Healthcare accessibility improvement
        - Anti-corruption measures
        - Democratic institution strengthening
        """)
    
    # Final insights box
    st.markdown("""
    <div class="insight-box" style="background-color: #2e6ba0; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4;">
    <h4>🎯 Key Takeaway</h4>
    Happiness is multidimensional and context-dependent, but certain universal factors emerge: 
    economic security (not wealth), strong social bonds, health, freedom, and trust in institutions. 
    The most successful countries balance economic prosperity with social cohesion and individual liberty.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()