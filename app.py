import asyncio
import streamlit as st
import pandas as pd
import re
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import time

# ========== Helper Functions ==========

async def fetch_page_content(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto(url, timeout=60000)
            content = await page.content()
        except Exception as e:
            print(f"Failed to load {url}: {e}")
            content = ""
        await browser.close()
        return content

def extract_clean_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Remove header, footer, nav, sidebar, breadcrumbs
    for tag in soup.find_all(['header', 'footer', 'nav', 'aside', 'form']):
        tag.decompose()
    for div in soup.find_all('div', class_=re.compile(r"breadcrumb", re.I)):
        div.decompose()

    # Remove headings
    for tag in soup.find_all(re.compile('^h[1-6]$')):
        tag.decompose()

    # Extract meaningful text
    paragraphs = soup.find_all('p')
    text = ' '.join(p.get_text(separator=' ', strip=True) for p in paragraphs)

    return text.strip()

def highlight_anchor(text, keyword):
    pattern = re.compile(rf"(.{{0,50}})({re.escape(keyword)})(.{{0,50}})", re.IGNORECASE)
    matches = pattern.findall(text)
    highlighted = []
    for before, match, after in matches:
        context = f"...{before}**[{match}]({target_url})**{after}..."
        highlighted.append(context)
    return highlighted

# ========== Streamlit UI ==========

st.set_page_config(page_title="Internal Linking Opportunity Finder", layout="wide")
st.title("Internal Linking Opportunity Finder")

uploaded_file = st.file_uploader("Upload a CSV or Excel file containing Page URLs", type=["csv", "xlsx"])
target_url = st.text_input("Enter the Target Page URL (where you want to build links):")
seed_keyword = st.text_input("Enter the Seed Keyword/Anchor Text:")

if uploaded_file and target_url and seed_keyword:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'URL' not in df.columns:
        st.error("Uploaded file must contain a 'URL' column!")
    else:
        urls = df['URL'].dropna().tolist()

        st.info(f"Found {len(urls)} pages to process.")

        progress = st.progress(0)
        page_contents = {}
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for idx, url in enumerate(urls):
            content = loop.run_until_complete(fetch_page_content(url))
            cleaned = extract_clean_content(content)
            page_contents[url] = cleaned
            progress.progress((idx + 1) / len(urls))
            time.sleep(0.2)

        # Fetch and clean target URL content
        st.info("Fetching target page content...")
        target_content_html = loop.run_until_complete(fetch_page_content(target_url))
        target_content = extract_clean_content(target_content_html)

        if not target_content:
            st.error("Failed to fetch target URL content.")
        else:
            st.success("Fetched all pages successfully!")

            st.info("Calculating semantic similarity...")
            pages_text = list(page_contents.values())
            all_texts = [target_content] + pages_text
            vectorizer = TfidfVectorizer().fit(all_texts)
            vectors = vectorizer.transform(all_texts)

            target_vec = vectors[0]
            page_vecs = vectors[1:]
            similarities = cosine_similarity(target_vec, page_vecs).flatten()

            results = []

            for url, text, score in zip(urls, pages_text, similarities):
                if score > 0.65 and re.search(seed_keyword, text, re.IGNORECASE):
                    # Avoid suggesting links if already present
                    if re.search(rf'href=["\'].*?{re.escape(target_url)}.*?["\']', text):
                        continue

                    anchors = highlight_anchor(text, seed_keyword)
                    if anchors:
                        results.append({
                            'Page URL': url,
                            'Similarity Score': round(score, 2),
                            'Suggested Placement (Context)': " | ".join(anchors)
                        })

            if not results:
                st.warning("No valid internal linking opportunities found!")
            else:
                output_df = pd.DataFrame(results)
                st.dataframe(output_df)

                tmp_download = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
                output_df.to_csv(tmp_download.name, index=False)

                st.download_button(
                    label="Download Opportunities CSV",
                    data=open(tmp_download.name, "rb").read(),
                    file_name="internal_linking_opportunities.csv",
                    mime="text/csv"
                )
