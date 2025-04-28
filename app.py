import streamlit as st
import pandas as pd
import httpx
from selectolax.parser import HTMLParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Internal Linking Finder", layout="wide")
st.title("🔗 Internal Linking Opportunity Finder")

# Helper: Fetch page content
def fetch_page_content(url):
    try:
        response = httpx.get(url, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.error(f"Failed to fetch {url}: {e}")
        return None

# Helper: Extract clean text (body only, skip header/footer/h1-h6)
def extract_main_content(html_content):
    tree = HTMLParser(html_content)
    
    # Remove header, footer, nav, and sidebars
    for selector in ['header', 'footer', 'nav', 'aside']:
        for node in tree.css(selector):
            node.decompose()

    # Remove all H1-H6 tags
    for i in range(1, 7):
        for node in tree.css(f'h{i}'):
            node.decompose()
    
    # Get body text
    body = tree.css_first('body')
    if body:
        text = body.text(separator=' ').strip()
        return re.sub(r'\s+', ' ', text)  # Clean multiple spaces
    else:
        return ""

# Helper: Find anchor opportunities
def find_anchor_opportunities(page_text, seed_keyword):
    # Split content into sentences
    sentences = re.split(r'(?<=[.!?])\s+', page_text)
    
    # Find sentences containing the seed keyword
    suggestions = []
    for sent in sentences:
        if seed_keyword.lower() in sent.lower():
            highlighted = sent.replace(seed_keyword, f"**{seed_keyword}**")
            suggestions.append((sent, highlighted))
    return suggestions

# Upload file
uploaded_file = st.file_uploader("Upload your CSV or Excel file containing URLs:", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success("File uploaded successfully!")

    url_column = st.selectbox("Select the column that contains URLs:", df.columns.tolist())
    target_url = st.text_input("Enter the Target URL:")
    seed_keyword = st.text_input("Enter the Seed Keyword:")

    if st.button("Find Internal Linking Opportunities"):
        progress = st.progress(0)
        results = []

        urls = df[url_column].dropna().unique().tolist()
        
        # Fetch target URL content separately to avoid self-linking
        target_page_html = fetch_page_content(target_url)
        target_page_text = extract_main_content(target_page_html) if target_page_html else ""

        for idx, url in enumerate(urls):
            progress.progress((idx + 1) / len(urls))
            if url.strip() == target_url.strip():
                continue  # Skip the target page itself
            
            html = fetch_page_content(url)
            if not html:
                continue
            page_text = extract_main_content(html)

            # Skip if no body text
            if not page_text:
                continue

            # Find opportunities
            suggestions = find_anchor_opportunities(page_text, seed_keyword)
            for original, highlighted in suggestions:
                results.append({
                    "Source URL": url,
                    "Suggested Sentence": highlighted,
                    "Anchor Text": seed_keyword,
                    "Target Link": target_url
                })

        if results:
            result_df = pd.DataFrame(results)
            st.success(f"Found {len(results)} internal link opportunities!")
            st.dataframe(result_df)

            # Download
            csv = result_df.to_csv(index=False)
            st.download_button("Download Results as CSV", csv, "internal_links.csv", "text/csv")
        else:
            st.warning("No internal linking opportunities found.")

