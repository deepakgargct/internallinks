import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

def fetch_text(url, retries=3, delay=1):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                time.sleep(delay)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unnecessary tags
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                tag.decompose()

            # Remove breadcrumb containers by common class names
            breadcrumb_classes = ['breadcrumb', 'breadcrumbs', 'woocommerce-breadcrumb']
            for bc in breadcrumb_classes:
                for tag in soup.find_all(class_=bc):
                    tag.decompose()

            text = soup.get_text(separator='\n', strip=True)
            return text
        except Exception:
            time.sleep(delay)
    return ""

def fetch_html(url, retries=3, delay=1):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                time.sleep(delay)
                continue
            return response.text
        except Exception:
            time.sleep(delay)
    return ""

def link_already_exists(html_content, target_url):
    # Check if the target URL already linked
    return target_url in html_content

def find_matching_sentences(text, keyword):
    sentences = re.split(r'(?<=[.!?]) +', text)
    matching_sentences = [sentence for sentence in sentences if keyword.lower() in sentence.lower()]
    return matching_sentences

def main():
    st.title("Internal Linking Opportunities Finder 🔗✨")

    uploaded_file = st.file_uploader("Upload your Pages List (CSV or Excel)", type=["csv", "xlsx"])
    target_url = st.text_input("Enter the Target URL (where you want to build links):")
    seed_keyword = st.text_input("Enter the Seed Keyword (anchor text):")

    if uploaded_file and target_url and seed_keyword:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'URL' not in df.columns:
            st.error("The uploaded file must contain a 'URL' column.")
            return

        page_urls = df['URL'].tolist()

        st.info(f"Found {len(page_urls)} pages. Crawling pages... ⏳")

        texts = []
        htmls = []
        valid_urls = []

        for url in page_urls:
            if url.strip() == target_url.strip():
                continue  # Skip the target page itself

            html_content = fetch_html(url)
            if not html_content:
                continue

            if link_already_exists(html_content, target_url):
                continue  # Skip if target link already exists

            text_content = fetch_text(url)
            if text_content:
                valid_urls.append(url)
                texts.append(text_content)
                htmls.append(html_content)

        if not texts:
            st.warning("No suitable pages found after crawling.")
            return

        st.success(f"Crawled {len(valid_urls)} valid pages. Now finding opportunities... 🔍")

        # Fetch content of the target page
        target_text = fetch_text(target_url)

        # Vectorize and calculate similarity
        tfidf = TfidfVectorizer(stop_words='english')
        vectors = tfidf.fit_transform([target_text] + texts)
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Find best matches
        results = []
        for idx, score in enumerate(similarities):
            if score > 0.25:  # Similarity threshold (adjustable)
                matches = find_matching_sentences(texts[idx], seed_keyword)
                for match in matches:
                    results.append({
                        'Source Page': valid_urls[idx],
                        'Matching Text': match.strip(),
                        'Anchor Text': seed_keyword,
                        'Target URL': target_url,
                        'Similarity Score': round(score, 3)
                    })

        if results:
            output_df = pd.DataFrame(results)
            st.dataframe(output_df)

            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Opportunities CSV",
                data=csv,
                file_name="internal_link_opportunities.csv",
                mime="text/csv"
            )
        else:
            st.warning("No internal linking opportunities found with the given seed keyword.")

if __name__ == "__main__":
    main()
