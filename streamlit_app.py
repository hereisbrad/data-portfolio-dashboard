import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(page_title="Brad Bitangane's Data Portfolio", 
                   page_icon="📊", layout="wide")

st.markdown("""
    <style>
    .main { 
        background-color: #f5f5f5; 
    }
    .css-18e3th9 {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Profile photo
st.sidebar.image("info\Photo_d'indentite.png", width=200)

st.sidebar.markdown("""
### About Me
I am a data science student with a focus on machine learning, data analytics, and optimization. 
I have experience working with real-world datasets, building models, and delivering insights that drive decision-making.
""")

# Download resume
with open("info\Brad_BItangane_Cv_Alternance_Analyst BI.pdf", "rb") as pdf_file:
    st.sidebar.download_button("Download My Resume", data=pdf_file, file_name="Brad_Bitangane_CV.pdf", mime="application/pdf")

# Dataset loading function
@st.cache_data
def load_data(data_name):
    if data_name == "Iris":
        return pd.read_csv('datasets/Iris.csv')
    elif data_name == "Titanic":
        return pd.read_csv('datasets/titanic.csv')
    elif data_name == "Red Wine Quality":
        return pd.read_csv('datasets/winequality_red.csv', delimiter=',')
    elif data_name == "White Wine Quality":
        return pd.read_csv('datasets/winequality_white.csv', delimiter=',')
    else:
        return None

# Data selection
st.sidebar.header("Select a Dataset")
dataset_name = st.sidebar.selectbox("Choose a dataset", 
                                    ["Iris", "Titanic", "Red Wine Quality", "White Wine Quality"])

data = load_data(dataset_name)

# Descriptive statistics
st.write(f"## {dataset_name} Dataset")
st.dataframe(data.head())
st.write(f"### Descriptive Statistics for {dataset_name} Dataset")
st.write(data.describe())

# Dataset download button
csv_data = data.to_csv(index=False)
st.sidebar.download_button(label="Download Dataset as CSV", data=csv_data, mime="text/csv")

# Iris Dataset
if dataset_name == "Iris":
    st.write("### Iris Dataset: How Do Flower Measurements Help Classify Species?")
    st.markdown("""
    This dataset represents the measurements of **three species** of iris flowers: **Setosa**, **Versicolor**, and **Virginica**. 
    These measurements include the lengths and widths of petals and sepals. The question we want to answer here is: 
    **How can flower measurements be used to differentiate between species?**
    """)

    # Sepal 
    st.write("#### Sepal Length vs Sepal Width")
    st.markdown("""
    To start, we look at how the **sepal length** and **sepal width** vary between species. These measurements capture the size of the flower.
    We use a scatter plot to visually explore whether these two measurements alone can help separate the species.
    """)
    fig = px.scatter(data, x='SepalLengthCm', y='SepalWidthCm', color='Species', 
                     title="Sepal Length vs Sepal Width by Species")
    st.plotly_chart(fig)
    
    # Petal 
    st.write("#### Petal Length vs Petal Width")
    st.markdown("""
    Next, we look at the **petal measurements**. Petals are often more distinctive than sepals, and we explore whether petal dimensions 
    offer clearer separation between species.
    """)
    fig = px.scatter(data, x='PetalLengthCm', y='PetalWidthCm', color='Species',
                     title="Petal Length vs Petal Width by Species")
    st.plotly_chart(fig)
    
    # Correlation heatmap
    st.write("#### Correlation Heatmap: Do Length and Width Correlate?")
    st.markdown("""
    Now that we’ve seen how the dimensions differ across species, we want to understand the relationships between the different measurements. 
    This **correlation heatmap** shows how strongly related features are, which can help us determine which measurements might be more useful for classification.
    """)
    corr_iris = data.drop(columns=["Id", "Species"]).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_iris, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.write("""
    **Insights:**
    - **Petal measurements** are much more correlated with each other than sepal measurements.
    - **Petal length** and **petal width** are key differentiators, particularly for distinguishing the **Virginica** species.
    """)

# Titanic Dataset
elif dataset_name == "Titanic":
    st.write("### Titanic Dataset: Survival Analysis")
    st.markdown("""
    The **Titanic dataset** is famous for providing data on passengers aboard the ship during its tragic voyage. The key question we explore is: 
    **Which factors contributed most to a passenger’s chances of survival?**
    """)

    # Survival by class and gender
    st.write("#### Survival Rate by Class and Gender")
    st.markdown("""
    **Class** and **gender** played critical roles in determining survival. First-class passengers often had access to lifeboats, and women were given priority. 
    Here, we use a bar chart to compare survival rates across **class** and **gender**.
    """)
    fig = px.histogram(data, x='Pclass', color='Sex', barmode='group', histfunc='count', facet_col='Survived',
                       title="Survival Rate by Class and Gender")
    st.plotly_chart(fig)

    # Age distribution
    st.write("#### Age Distribution of Passengers")
    st.markdown("""
    Age also influenced survival, with children being prioritized. This plot shows the distribution of passenger ages and helps us 
    understand the demographic of the passengers.
    """)
    fig = px.histogram(data, x='Age', nbins=20, title="Age Distribution of Titanic Passengers")
    st.plotly_chart(fig)
    
    st.write("""
    **Insights:**
    - **Women in first class** had the highest survival rates, while passengers in third class (mostly male) had lower chances of survival.
    - There were many children aboard, but survival among children was not as high as might be expected.
    """)

    # Fare distribution by class and survival
    st.write("#### Fare Distribution by Class and Survival")
    st.markdown("""
    Wealthier passengers paid higher **fares**, and these fares were often associated with better survival chances, as they had access to lifeboats. 
    This box plot shows the relationship between **fare**, **class**, and **survival** status.
    """)
    fig = px.box(data, x='Pclass', y='Fare', color='Survived', title="Fare Distribution by Class and Survival")
    st.plotly_chart(fig)
    
    st.write("""
    **Insights:**
    - Higher fares were closely tied to **first-class passengers**, who had significantly better survival outcomes.
    """)

# Red Wine Quality Dataset
elif dataset_name == "Red Wine Quality":
    st.write("### Red Wine Quality: What Affects Wine Quality?")
    st.markdown("""
    This dataset explores the chemical properties of red wine and their relationship to **quality ratings**. 
    Our goal is to identify which chemical characteristics influence the perceived quality of wine.
    """)

    # Wine quality distribution
    st.write("#### Distribution of Red Wine Quality Scores")
    st.markdown("""
    We start by looking at the distribution of wine quality ratings. How often do wines receive the highest scores, and are most wines of average quality?
    """)
    fig = px.histogram(data, x='quality', nbins=10, title="Red Wine Quality Distribution")
    st.plotly_chart(fig)
    
    st.write("""
    **Insights:**
    - Most wines score between **5 and 7**, indicating that the highest-quality wines are relatively rare.
    """)

    # Correlation heatmap
    st.write("#### Correlation Heatmap: Chemical Properties vs. Quality")
    st.markdown("""
    This heatmap shows the relationship between various chemical properties, like **alcohol content** and **acidity**, and the quality score of each wine.
    Understanding these correlations helps us pinpoint which factors most influence wine quality.
    """)
    corr_red_wine = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_red_wine, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
    st.write("""
    **Insights:**
    - **Alcohol content** is positively correlated with higher quality ratings, meaning wines with higher alcohol tend to be rated better.
    - **Volatile acidity** is negatively correlated with quality, indicating that high acidity detracts from the perceived quality of the wine.
    """)

# White Wine Quality Dataset
elif dataset_name == "White Wine Quality":
    st.write("### White Wine Quality: What Affects White Wine Quality?")
    st.markdown("""
    This dataset analyzes the chemical properties of white wines and their corresponding **quality ratings**. 
    Our goal is to understand which factors most impact the taste and quality of white wines.
    """)

    # White wine quality distribution
    st.write("#### Distribution of White Wine Quality Scores")
    st.markdown("""
    We begin by examining the distribution of quality scores for white wine. This will help us understand how common high-quality white wines are.
    """)
    fig = px.histogram(data, x='quality', nbins=10, title="White Wine Quality Distribution")
    st.plotly_chart(fig)

    st.write("""
    **Insights:**
    - Similar to red wines, most white wines score between **5 and 7**, suggesting that exceptionally high-quality wines are rare.
    """)

    # Correlation heatmap
    st.write("#### Correlation Heatmap: Chemical Properties vs. Quality")
    st.markdown("""
    This heatmap helps us explore the relationship between different chemical properties, such as **residual sugar**, **alcohol content**, and **acidity**, 
    and how they influence the quality rating of white wine.
    """)
    corr_white_wine = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_white_wine, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.write("""
    **Insights:**
    - **Residual sugar** has a more significant positive correlation with quality for white wine compared to red wine, which may explain the sweeter nature of higher-quality white wines.
    - **Alcohol content** continues to play a role in determining quality, with higher alcohol levels generally leading to higher ratings.
    """)

# Footer
st.markdown("---")
st.markdown("Created by Brad Bitangane as part of my professional data portfolio using Streamlit.")

