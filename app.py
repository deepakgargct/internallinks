import streamlit as st
import pandas as pd
import asyncio
from playwright.async_api import async_playwright
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="Internal Linking Opportunity Finder", layout="wide")
st.title("\ud83c\udf10 Internal Linking Opportunity Finder")

# ---- Helper Functions ----
async def fetch_page_content(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=60000)
            html = await page.content()
            await browser.close()

            soup = BeautifulSoup(html, 'html.parser')

            # Remove header, footer, nav, aside, H1-H6, breadcrumbs
            for tag in soup(['header', 'footer', 'nav', 'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                tag.decompose()

            # Also remove common breadcrumb classes
            breadcrumb_classes = ['breadcrumb', 'breadcrumbs', 'site-breadcrumb']
            for cls in breadcrumb_classes:
                for tag in soup.select(f'.{cls}'):
                    tag.decompose()

            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)
            else:
                return ""
    except Exception as e:
        return ""

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

async def crawl_all_pages(urls):
    results = {}
    total = len(urls)
    count = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for url in urls:
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=60000)
                html = await page.content()

                soup = BeautifulSoup(html, 'html.parser')
                for tag in soup(['header', 'footer', 'nav', 'aside', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    tag.decompose()
                breadcrumb_classes = ['breadcrumb', 'breadcrumbs', 'site-breadcrumb']
                for cls in breadcrumb_classes:
                    for tag in soup.select(f'.{cls}'):
                        tag.decompose()

                body = soup.find('body')
                if body:
                    text = body.get_text(separator=' ', strip=True)
                    results[url] = re.sub(r'\s+', ' ', text)
            except Exception as e:
                results[url] = ""
            count += 1
            st.progress(count / total)
        await browser.close()
    return results

# ---- Main App ----
uploaded_file = st.file_uploader("Upload your CSV or Excel with URLs", type=['csv', 'xlsx'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if 'url' not in df.columns:
        st.error("File must contain a 'url' column.")
    else:
        urls = df['url'].dropna().tolist()

        target_url = st.text_input("Enter Target URL:")
        seed_keyword = st.text_input("Enter Seed Keyword/Anchor Text:")

        if st.button("Find Internal Linking Opportunities"):
            with st.spinner("Fetching pages..."):
                target_content = asyncio.run(fetch_page_content(target_url))
                if not target_content:
                    st.error("Failed to fetch target URL content.")
                else:
                    pages_content = asyncio.run(crawl_all_pages(urls))

                    # Remove pages with no content
                    pages_content = {url: text for url, text in pages_content.items() if text.strip()}

                    if not pages_content:
                        st.error("No valid pages found for internal linking.")
                    else:
                        all_texts = [target_content] + list(pages_content.values())
                        vectorizer = TfidfVectorizer(stop_words='english')
                        tfidf_matrix = vectorizer.fit_transform(all_texts)

                        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

                        results = []
                        for idx, (url, content) in enumerate(pages_content.items()):
                            if seed_keyword.lower() in content.lower():
                                # Check if already linking
                                if target_url not in content:
                                    similarity = similarities[idx]
                                    sentences = re.split(r'(?<=[.!?]) +', content)
                                    best_snippet = ""
                                    for sentence in sentences:
                                        if seed_keyword.lower() in sentence.lower():
                                            best_snippet = sentence.strip()
                                            break
                                    if best_snippet:
                                        results.append({
                                            'Source Page': url,
                                            'Suggested Snippet': best_snippet,
                                            'Anchor Text': seed_keyword,
                                            'Target Link': target_url,
                                            'Similarity Score': round(float(similarity), 3)
                                        })

                        if results:
                            output_df = pd.DataFrame(results)
                            st.success(f"Found {len(results)} internal linking opportunities!")
                            st.dataframe(output_df)

                            csv = output_df.to_csv(index=False)
                            st.download_button("\ud83d\udd17 Download Results", csv, file_name="internal_linking_opportunities.csv", mime="text/csv")
                        else:
                            st.warning("No good internal linking opportunities found.")
