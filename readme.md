
````markdown
# 🌍 World Happiness Index Analysis Dashboard

A powerful and interactive dashboard built using **Streamlit** and **Plotly** to analyze the **World Happiness Report** over the years. The app enables users to explore global trends, compare countries and regions, and understand the key factors contributing to happiness across the world.

---

## 📌 Features

- 📊 **Overview & Global Trends**: Visualize global happiness metrics and their evolution over time.
- 🏆 **Country Rankings**: See annual rankings of countries by happiness score.
- 🔍 **Country Deep Dive**: Explore detailed trends and metrics for individual countries.
- ⚖️ **Country Comparison**: Compare multiple countries across various happiness factors.
- 🌎 **Regional Analysis**: Analyze average happiness scores and factor contributions by region.
- 📈 **Factor Analysis**: Examine how different factors like GDP, freedom, and corruption correlate with happiness.
- 🔮 **Insights & Predictions**: View actionable insights and predictive trends from data.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Backend/Data**: Pandas, NumPy, SciPy
- **Data Source**: World Happiness Report (2015–2024) or synthetic data when not available

---

## 🧑‍💻 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/world-happiness-dashboard.git
cd world-happiness-dashboard
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Dataset

Place your `WHRFinal.xlsx` file in the `Data/` directory.

> If the dataset is missing, the app will generate sample data automatically.

### 4. Run the Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
.
├── app.py                  # Main Streamlit application
├── Data/
│   └── WHRFinal.xlsx       # World Happiness Report dataset
├── README.md               # Project documentation
├── factor-analysis.md      # Factor analysis
├── requirements.txt        # Python dependencies
```

---

## 📊 Dataset Columns

The app works with a dataset having the following columns:

* `Country`
* `Year`
* `Region`
* `Happiness Rank`
* `Happiness Score`
* `GDP`
* `Social Support`
* `Life Expectancy`
* `Freedom`
* `Generosity`
* `Corruption`

---

## ✨ Visualizations & UI

* Interactive charts and line plots via **Plotly**
* Clean layout with styled metrics and sidebar filters
* Custom CSS for polished UI appearance

---

## 📎 Example Insights

* Countries with stronger social support and healthcare systems tend to rank higher.
* High GDP doesn't always mean high happiness — freedom and trust in government matter.
* Nordic countries consistently lead the happiness rankings.

---

## 📃 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* World Happiness Report: [https://worldhappiness.report/](https://worldhappiness.report/)
* Streamlit for rapid dashboard development
* Plotly for interactive visualizations


