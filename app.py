import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

def get_clean_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'form']):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

def find_internal_link_opportunities(url_list, target_url, keyword, threshold=0.65):
    url_list = [url for url in url_list if url != target_url]
    target_text = get_clean_text(target_url)

    if not target_text:
        st.error("Failed to fetch content from the target URL.")
        return []

    candidates = []
    texts = []

    for url in url_list:
        text = get_clean_text(url)
        if keyword.lower() in text.lower() and len(text) > 100:
            candidates.append(url)
            texts.append(text)

    if not texts:
        return []

    all_texts = [target_text] + texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked = [(candidates[i], similarities[i]) for i in range(len(candidates))]
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    return [(url, score) for url, score in ranked if score > threshold]

# ---------------- Streamlit App ----------------------

st.title("🔗 Internal Linking Opportunities Finder (Upload Version)")

uploaded_file = st.file_uploader("Upload CSV or Excel file with a column named 'url'", type=["csv", "xlsx"])
target_url = st.text_input("Enter the Target URL (e.g. https://example.com/guide)")
keyword = st.text_input("Enter the Seed Keyword (e.g. contact lenses)")

if uploaded_file and target_url and keyword:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'url' not in df.columns:
            st.error("File must contain a column named 'url'")
        else:
            url_list = df['url'].dropna().unique().tolist()
            with st.spinner("Analyzing internal pages..."):
                results = find_internal_link_opportunities(url_list, target_url, keyword)

            if results:
                st.success(f"Found {len(results)} internal linking opportunities.")
                result_df = pd.DataFrame(results, columns=["URL", "Similarity Score"])
                st.dataframe(result_df)

                # Download button
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                st.download_button("Download Results as CSV", csv_buffer.getvalue(), "internal_links.csv", "text/csv")
            else:
                st.info("No relevant internal linking opportunities found.")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Please upload a file and fill in all fields.")
