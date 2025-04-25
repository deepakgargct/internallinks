import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Internal Linking Recommender", layout="wide")
st.title("🔗 Internal Linking Opportunity Finder (SEO)")

# --- Function to fetch page text with retries and delay
def fetch_text(url, retries=3, delay=1):
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                st.warning(f"Attempt {attempt+1}: Failed to fetch {url} — Status: {response.status_code}")
                time.sleep(delay)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            if len(text) < 100:
                st.warning(f"⚠️ Very short content from {url} — might be empty or blocked.")
            return text

        except Exception as e:
            st.warning(f"Attempt {attempt+1}: Error fetching {url}: {str(e)}")
            time.sleep(delay)

    st.error(f"❌ Failed to fetch {url} after {retries} attempts.")
    return ""

# --- Upload file
uploaded_file = st.file_uploader("📄 Upload CSV or Excel file with page URLs", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if df.columns[0] != 'URL':
        st.error("🚫 The file must have a column named 'URL'")
    else:
        urls = df['URL'].dropna().tolist()

        target_url = st.text_input("🔗 Enter the **Target URL** (the page you want internal links to)")
        seed_keyword = st.text_input("🔍 Enter the **Seed Keyword** or Anchor Text (e.g. 'contact lenses')")

        if target_url and seed_keyword and st.button("Find Linking Opportunities"):
            st.info("🔄 Crawling pages and analyzing content. This may take a minute...")

            # Fetch content for all pages
            contents = []
            valid_urls = []

            for url in urls:
                if url.strip() == target_url.strip():
                    continue  # Skip the target URL
                text = fetch_text(url)
                if text:
                    contents.append(text)
                    valid_urls.append(url)

            # Fetch content for target URL
            target_text = fetch_text(target_url)

            # TF-IDF similarity
            corpus = [target_text] + contents
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(corpus)
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

            results = []
            for i, sim in enumerate(cosine_sim):
                if seed_keyword.lower() in contents[i].lower():
                    results.append({
                        'URL': valid_urls[i],
                        'Similarity Score': round(sim, 3)
                    })

            results = sorted(results, key=lambda x: x['Similarity Score'], reverse=True)

            if results:
                st.success(f"✅ Found {len(results)} internal linking opportunities.")
                st.dataframe(results[:10], use_container_width=True)
            else:
                st.warning("🤷 No relevant pages found with keyword and similarity match.")
