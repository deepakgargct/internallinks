import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Internal Linking Recommender", layout="wide")
st.title("🔗 Internal Linking Opportunity Finder (SEO)")

# --- Function to fetch clean text content from a URL
def fetch_text(url, retries=3, delay=1):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                time.sleep(delay)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()

            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception:
            time.sleep(delay)
    return ""

# --- Upload file
uploaded_file = st.file_uploader("📄 Upload CSV or Excel file with a column named 'URL'", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if df.columns[0] != 'URL':
        st.error("🚫 The file must have a column named 'URL'")
    else:
        urls = df['URL'].dropna().tolist()

        target_url = st.text_input("🔗 Enter the **Target URL**")
        seed_keyword = st.text_input("🔍 Enter the **Seed Keyword** (e.g. 'contact lenses')")

        if target_url and seed_keyword and st.button("Find Internal Link Opportunities"):
            st.info("🔄 Crawling and analyzing...")

            # Fetch content for all candidate pages
            contents = []
            valid_urls = []

            for url in urls:
                if url.strip() == target_url.strip():
                    continue
                text = fetch_text(url)
                if text and len(text.split()) > 50:  # Basic check for minimal content
                    contents.append(text)
                    valid_urls.append(url)

            # Fetch content of the target page
            target_text = fetch_text(target_url)

            if not target_text or len(contents) == 0:
                st.error("❌ Unable to retrieve enough content for analysis.")
            else:
                # TF-IDF vectorization
                corpus = [target_text] + contents
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(corpus)
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

                results = []
                keyword_lower = seed_keyword.lower()

                for i, sim in enumerate(cosine_sim):
                    content = contents[i]
                    if keyword_lower in content.lower():
                        # Extract matching sentence/line
                        matching_lines = [line.strip() for line in content.split('.') if keyword_lower in line.lower()]
                        suggested_line = matching_lines[0] if matching_lines else "Keyword found but context not extractable."

                        results.append({
                            'URL': valid_urls[i],
                            'Similarity Score': round(sim, 3),
                            'Suggested Placement': suggested_line
                        })

                if results:
                    results = sorted(results, key=lambda x: x['Similarity Score'], reverse=True)
                    result_df = pd.DataFrame(results[:10])

                    st.success(f"✅ Found {len(result_df)} opportunities.")
                    st.dataframe(result_df, use_container_width=True)

                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name='internal_link_opportunities.csv',
                        mime='text/csv'
                    )
                else:
                    st.warning("⚠️ No relevant matches found with both similarity and keyword.")
