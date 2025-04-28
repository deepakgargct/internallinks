import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

def fetch_clean_text(url, retries=3, delay=1):
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                time.sleep(delay)
                continue

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted tags
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                tag.decompose()

            breadcrumb_classes = ['breadcrumb', 'breadcrumbs', 'woocommerce-breadcrumb']
            for bc in breadcrumb_classes:
                for tag in soup.find_all(class_=bc):
                    tag.decompose()

            paragraphs = []
            for p in soup.find_all(['p', 'div', 'span']):
                text = p.get_text(separator=' ', strip=True)
                if text and len(text.split()) > 5:  # Ignore tiny fragments
                    paragraphs.append(text)

            full_text = "\n".join(paragraphs)
            return full_text
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

def link_exists_in_html(html_content, target_url):
    return target_url in html_content

def find_matching_sentences(text, keyword):
    sentences = re.split(r'(?<=[.!?]) +', text)
    matches = []
    for sentence in sentences:
        if keyword.lower() in sentence.lower():
            highlighted = re.sub(f"({re.escape(keyword)})", r"[\1]", sentence, flags=re.I)
            matches.append(highlighted)
    return matches

def main():
    st.title("🔗 Smart Internal Linking Tool — Pro Version")

    uploaded_file = st.file_uploader("Upload your Pages List (CSV or Excel)", type=["csv", "xlsx"])
    target_url = st.text_input("Enter the Target URL (where you want to add internal links):")
    seed_keyword = st.text_input("Enter the Seed Keyword (Anchor Text):")

    if uploaded_file and target_url and seed_keyword:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'URL' not in df.columns:
            st.error("The uploaded file must have a 'URL' column.")
            return

        page_urls = df['URL'].tolist()

        st.info(f"Crawling {len(page_urls)} pages... Please wait ⏳")

        source_texts = []
        page_htmls = []
        valid_urls = []

        for url in page_urls:
            if url.strip() == target_url.strip():
                continue  # Skip Target Page

            html_content = fetch_html(url)
            if not html_content:
                continue

            if link_exists_in_html(html_content, target_url):
                continue  # Skip if link already exists

            clean_text = fetch_clean_text(url)
            if clean_text:
                valid_urls.append(url)
                source_texts.append(clean_text)
                page_htmls.append(html_content)

        if not source_texts:
            st.warning("No valid pages found for internal linking.")
            return

        st.success(f"Analyzing {len(valid_urls)} valid pages for opportunities 🔍")

        # Fetch target page content
        target_text = fetch_clean_text(target_url)

        # Calculate semantic similarity
        tfidf = TfidfVectorizer(stop_words='english')
        vectors = tfidf.fit_transform([target_text] + source_texts)
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Build results
        results = []
        for idx, score in enumerate(similarities):
            if score > 0.25:  # Adjust threshold as needed
                matched_sentences = find_matching_sentences(source_texts[idx], seed_keyword)
                for match in matched_sentences:
                    results.append({
                        'Source Page': valid_urls[idx],
                        'Matching Text (highlighted)': match.strip(),
                        'Suggested Anchor Text': seed_keyword,
                        'Link to Target Page': target_url,
                        'Similarity Score': round(score, 3)
                    })

        if results:
            output_df = pd.DataFrame(results)
            st.dataframe(output_df)

            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Internal Link Opportunities CSV",
                data=csv,
                file_name="internal_link_suggestions.csv",
                mime="text/csv"
            )
        else:
            st.warning("No good internal linking opportunities found.")

if __name__ == "__main__":
    main()
