import pandas as pd

# Load dataset
df = pd.read_csv("metadata.csv") 

# Check shape and columns
print(f"Shape of dataset: {df.shape}\n")
print("Columns in the dataset:")
print(df.columns.tolist())

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Check missing values
missing_count = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    'missing_count': missing_count,
    'missing_percent': missing_percent
}).sort_values(by='missing_percent', ascending=False)

print(missing_summary)

# Drop columns with too many missing values
cols_to_drop = ['mag_id', 'arxiv_id', 'pmc_json_files', 'pdf_json_files', 'sha']
df_clean = df.drop(columns=cols_to_drop)

print("Columns after dropping high-missing ones:", df_clean.columns.tolist())
# Drop rows with missing abstracts or publish_time
df_clean = df_clean.dropna(subset=['abstract', 'publish_time']).copy()
print("Cleaned dataset shape:", df_clean.shape)

# 5️⃣ Convert publish_time to datetime and extract year
# ------------------------------
df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
df_clean = df_clean.dropna(subset=['publish_time'])  # drop rows that couldn't convert
df_clean['year'] = df_clean['publish_time'].dt.year.astype('Int32')

# ------------------------------
# 6️⃣ Create abstract word count column
# ------------------------------
df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))

# ------------------------------
# 7️⃣ Save the cleaned dataset
# ------------------------------
df_clean.to_csv(r"C:\Users\EUNICE\Desktop\PLP.PYTHON\Frameworks_Assignment\cord19_cleaned.csv", index=False)
print("Cleaned dataset saved as 'cord19_cleaned.csv'")

# ------------------------------
# ✅ Quick check
# ------------------------------
print("\nFirst 5 rows of cleaned data:\n", df_clean.head())
print("\nColumns and types:\n", df_clean.dtypes)
print("\nMissing values after cleaning:\n", df_clean.isnull().sum())

import pandas as pd

# Load cleaned dataset
df = pd.read_csv("cord19_cleaned.csv")

# Ensure 'publish_time' is datetime
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')

# Extract year from publication date
df['year'] = df['publish_time'].dt.year.astype('Int32')

# Create a new column for abstract word count
df['abstract_word_count'] = df['abstract'].apply(lambda x: len(str(x).split()))

# Quick check
print(df[['publish_time', 'year', 'abstract_word_count']].head())

import pandas as pd
from collections import Counter
import re

# Load the cleaned dataset
df = pd.read_csv("cord19_cleaned.csv", low_memory=False)

# Ensure 'publish_time' is datetime and 'year' exists
df['publish_time'] = pd.to_datetime(df['publish_time'], errors='coerce')
df['year'] = df['publish_time'].dt.year

# 1️⃣ Count papers by publication year
papers_by_year = df['year'].value_counts().sort_index()
print("Papers by year:")
print(papers_by_year)

# 2️⃣ Identify top journals publishing COVID-19 research
top_journals = df['journal'].value_counts().head(10)
print("\nTop 10 journals by number of papers:")
print(top_journals)

# 3️⃣ Find most frequent words in titles
# Combine all titles into a single string
all_titles = " ".join(df['title'].dropna()).lower()

# Remove punctuation and split into words
words = re.findall(r'\b\w+\b', all_titles)

# Count word frequencies
word_counts = Counter(words)

# Remove common stop words (optional)
stop_words = {'the', 'and', 'of', 'in', 'on', 'for', 'to', 'with', 'a', 'an', 'by', 'using', 'from', 'at'}
filtered_words = {word: count for word, count in word_counts.items() if word not in stop_words}

# Get the 20 most common words
most_common_words = Counter(filtered_words).most_common(20)
print("\nMost frequent words in titles:")
for word, count in most_common_words:
    print(f"{word}: {count}")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

# Load cleaned dataset
df = pd.read_csv("cord19_cleaned.csv", parse_dates=['publish_time'])

# --- 1️⃣ Number of publications over time ---
papers_by_year = df['year'].value_counts().sort_index()
plt.figure(figsize=(12,6))
sns.lineplot(x=papers_by_year.index, y=papers_by_year.values, marker='o')
plt.title("Number of COVID-19 Papers by Year")
plt.xlabel("Year")
plt.ylabel("Number of Papers")
plt.tight_layout()
plt.savefig("papers_by_year.png")
plt.close()

# --- 2️⃣ Top publishing journals ---
top_journals = df['journal'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=top_journals.values, y=top_journals.index, palette="viridis")
plt.title("Top 10 Journals by Number of COVID-19 Papers")
plt.xlabel("Number of Papers")
plt.ylabel("Journal")
plt.tight_layout()
plt.savefig("top_journals.png")
plt.close()

# --- 3️⃣ Word cloud of paper titles ---
text = " ".join(df['title'].dropna().astype(str).tolist())
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=1200, height=600, background_color='white', stopwords=stopwords).generate(text)
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig("title_wordcloud.png")
plt.close()

# --- 4️⃣ Distribution of paper counts by source ---
source_counts = df['source_x'].value_counts()
plt.figure(figsize=(8,6))
sns.barplot(x=source_counts.index, y=source_counts.values, palette="coolwarm")
plt.title("Distribution of Papers by Source")
plt.xlabel("Source")
plt.ylabel("Number of Papers")
plt.tight_layout()
plt.savefig("papers_by_source.png")
plt.close()

print("All visualizations saved as PNG files.")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

st.title("CORD-19 Data Explorer")
st.write("Interactive exploration of COVID-19 research papers")

# --- Load data safely ---
try:
    df = pd.read_csv("cord19_cleaned.csv", parse_dates=['publish_time'])
except FileNotFoundError:
    st.error("❌ Could not find 'cord19_cleaned.csv'. Please make sure the file is in the same folder.")
    st.stop()

# --- Debug: Show columns ---
st.write("### Columns in dataset:")
st.write(df.columns.tolist())

# --- Quick preview ---
st.subheader("Preview of Dataset")
st.dataframe(df.head())

# --- Sidebar filters ---
# --- Sidebar filters ---
st.sidebar.header("Filters")

if "year" not in df.columns:
    st.error("⚠️ The dataset does not have a 'year' column. Please check your CSV.")
    st.stop()

# Get min and max year safely
available_years = sorted(df['year'].dropna().unique().tolist())
min_year = int(min(available_years))
max_year = int(max(available_years))

# Show available years for debugging
st.sidebar.write("Available years in dataset:", available_years)

# Year range slider (dynamic default)
year_range = st.sidebar.slider(
    "Select year range",
    min_year,
    max_year,
    (min_year, max_year)  # default = full range
)

# Journal filter
journals = df['journal'].dropna().unique().tolist()
selected_journal = st.sidebar.selectbox("Select Journal", options=["All"] + journals)

# --- Apply filters ---
filtered_df = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
if selected_journal != "All":
    filtered_df = filtered_df[filtered_df['journal'] == selected_journal]

if filtered_df.empty:
    st.warning("⚠️ No papers found for the selected filters.")
    st.stop()

# --- Plot number of publications over time ---
st.subheader("Number of Publications Over Time")
papers_by_year = filtered_df.groupby('year').size()
fig, ax = plt.subplots()
sns.barplot(x=papers_by_year.index, y=papers_by_year.values, palette="viridis", ax=ax)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Papers")
st.pyplot(fig)

# --- Top journals ---
st.subheader("Top Publishing Journals")
top_journals = filtered_df['journal'].value_counts().head(10)
fig2, ax2 = plt.subplots()
sns.barplot(x=top_journals.values, y=top_journals.index, palette="coolwarm", ax=ax2)
ax2.set_xlabel("Number of Papers")
ax2.set_ylabel("Journal")
st.pyplot(fig2)

# --- Word cloud of titles ---
if "title" in filtered_df.columns:
    st.subheader("Word Cloud of Paper Titles")
    text = " ".join(filtered_df['title'].dropna().astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, stopwords=STOPWORDS, background_color="white").generate(text)
    fig3, ax3 = plt.subplots(figsize=(12,6))
    ax3.imshow(wordcloud, interpolation="bilinear")
    ax3.axis("off")
    st.pyplot(fig3)

# --- Distribution of papers by source ---
if 'source_x' in filtered_df.columns:
    st.subheader("Distribution of Papers by Source")
    source_counts = filtered_df['source_x'].value_counts().head(10)
    fig4, ax4 = plt.subplots()
    sns.barplot(x=source_counts.values, y=source_counts.index, palette="magma", ax=ax4)
    ax4.set_xlabel("Number of Papers")
    ax4.set_ylabel("Source")
    st.pyplot(fig4)

# --- Final data preview ---
st.subheader("Filtered Dataset Sample")
cols_to_show = [c for c in ["title", "authors", "journal", "publish_time", "abstract"] if c in filtered_df.columns]
st.dataframe(filtered_df[cols_to_show].head(10))

# ------------------------------------------------------
# Part 5: Documentation & Reflection
# ------------------------------------------------------

"""
📌 Documentation

This project explored the CORD-19 dataset (metadata.csv), focusing on metadata analysis
of COVID-19 research papers.

Steps Performed:
1. Data Loading
   - Loaded metadata.csv into a Pandas DataFrame.
   - Inspected dataset shape, columns, and missing values.

2. Data Cleaning
   - Dropped columns with excessive missing values (mag_id, arxiv_id, etc.).
   - Removed rows without abstracts or publish dates.
   - Converted publish_time to datetime and extracted year.
   - Added derived column 'abstract_word_count'.
   - Saved cleaned dataset as cord19_cleaned.csv.

3. Exploratory Analysis
   - Publications per year: trend of research papers.
   - Top journals: most frequent publishers.
   - Word cloud: common keywords in paper titles.
   - Distribution by source: dataset origin.

4. Visualization
   - Bar plots, line charts, and word clouds created with Matplotlib, Seaborn, and WordCloud.

5. Streamlit App
   - Built an interactive dashboard with sidebar filters (year range, journal).
   - Displayed charts and filtered dataset previews.

📑 Reflection

Challenges:
- Handling missing values consistently across columns.
- Large dataset size caused performance issues (e.g., Streamlit slider freezing).

Learnings:
- Improved skills in Pandas for cleaning and transformation.
- Learned to create multiple types of visualizations.
- Gained experience in building an interactive Streamlit app.

Next Steps:
- Add NLP-based keyword extraction and topic modeling.
- Analyze citations and collaborations.
- Optimize Streamlit performance for large datasets.
"""
