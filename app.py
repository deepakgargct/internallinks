import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import time

def fetch_rendered_html(url):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=60000)  # 60 seconds
            page.wait_for_load_state('networkidle')
            content = page.content()
            browser.close()
            return content
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def fetch_clean_text_from_rendered_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove unwanted sections
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        tag.decompose()

    breadcrumb_classes = ['breadcrumb', 'breadcrumbs', 'woocommerce-breadcrumb']
    for bc in breadcrumb_classes:
        for tag in soup.find_all(class_=bc):
            tag.decompose()

    paragraphs = []
    for p in soup.find_all(['p', 'div', 'span']):
        text = p.get_text(separator=' ', strip=True)
        if text and len(text.split()) > 5:  # Ignore short blocks
            paragraphs.append(text)

    return "\n".join(paragraphs)

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
    st.title("🔗 Internal Linking Tool — Pro Rendered Version")

    uploaded_file = st.file_uploader("Upload your Pages List (CSV or Excel)", type=["csv", "xlsx"])
    target_url = st.text_input("Enter the Target URL:")
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
        valid_urls = []

        for url in page_urls:
            if url.strip() == target_url.strip():
                continue  # Skip Target Page

            html_content = fetch_rendered_html(url)
            if not html_content:
                continue

            if link_exists_in_html(html_content, target_url):
                continue  # Skip if link already exists

            clean_text = fetch_clean_text_from_rendered_html(html_content)
            if clean_text:
                valid_urls.append(url)
                source_texts.append(clean_text)

        if not source_texts:
            st.error("❌ No valid pages found for internal linking (after full rendering).")
            return

        st.success(f"Analyzing {len(valid_urls)} valid pages ✅")

        # Fetch target page
        target_html = fetch_rendered_html(target_url)
        target_text = fetch_clean_text_from_rendered_html(target_html)

        # Semantic similarity
        tfidf = TfidfVectorizer(stop_words='english')
        vectors = tfidf.fit_transform([target_text] + source_texts)
        similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Build results
        results = []
        for idx, score in enumerate(similarities):
            if score > 0.25:
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
                label="Download Internal Link Suggestions CSV",
                data=csv,
                file_name="internal_link_suggestions.csv",
                mime="text/csv"
            )
        else:
            st.warning("No strong internal linking opportunities found.")

if __name__ == "__main__":
    main()
