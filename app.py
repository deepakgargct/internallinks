import streamlit as st
import pandas as pd
import httpx
from selectolax.parser import HTMLParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import urlparse
import re
import asyncio

# Function to fetch page content
async def fetch_content(url):
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
        except Exception as e:
            return None
    return None

# Function to clean body text (remove headers, footers, H1-H6)
def extract_main_text(html):
    tree = HTMLParser(html)

    # Remove unwanted sections
    for tag in tree.css('header, footer, nav, h1, h2, h3, h4, h5, h6, script, style'):
        tag.decompose()

    body = tree.css_first('body')
    if body:
        return body.text(separator="\n").strip()
    return ""

# Function to check if keyword already exists
def keyword_exists(html, keyword, target_url):
    tree = HTMLParser(html)
    for link in tree.css('a'):
        href = link.attributes.get('href', '')
        if target_url in href and keyword.lower() in link.text().lower():
            return True
    return False

# Streamlit App
st.title("🔗 Internal Linking Finder (Async Version)")

uploaded_file = st.file_uploader("Upload CSV or Excel with URLs", type=["csv", "xlsx"])
target_url = st.text_input("Enter Target Page URL (where links should point)")
seed_keyword = st.text_input("Enter Seed Keyword / Anchor Text")

if uploaded_file and target_url and seed_keyword:
    # Read URLs
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    urls = df.iloc[:, 0].dropna().tolist()
    urls = list(set(urls))  # Remove duplicates

    st.write(f"✅ {len(urls)} URLs loaded.")

    async def process_urls():
        target_html = await fetch_content(target_url)
        if not target_html:
            st.error("Failed to fetch target URL content.")
            return
        
        target_text = extract_main_text(target_html)

        pages_data = []
        texts = []

        for url in urls:
            if url == target_url:
                continue
            page_html = await fetch_content(url)
            if not page_html:
                continue

            if keyword_exists(page_html, seed_keyword, target_url):
                continue  # Skip if already linked

            page_text = extract_main_text(page_html)
            if len(page_text) > 50:  # Ignore very short pages
                pages_data.append((url, page_text, page_html))
                texts.append(page_text)

        if not pages_data:
            st.error("❗ No valid pages found for internal linking.")
            return

        # TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([target_text] + texts)
        cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        # Rank pages
        results = []
        for idx, (score, (url, text, html)) in enumerate(zip(cosine_similarities, pages_data)):
            if score > 0.35:  # Adjust threshold
                # Find suggested text snippet
                sentences = text.split(".")
                match = ""
                for sentence in sentences:
                    if seed_keyword.lower() in sentence.lower():
                        match = sentence.strip()
                        break

                if match:
                    results.append({
                        "Source Page": url,
                        "Suggested Sentence for Link": match,
                        "Similarity Score": round(score, 3)
                    })

        if results:
            output_df = pd.DataFrame(results)
            st.dataframe(output_df)

            # 🛠 Corrected download_button with proper parenthesis
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="internal_link_suggestions.csv",
                mime="text/csv"
            )
        else:
            st.warning("⚡ No strong matches found for internal linking.")

    asyncio.run(process_urls())
