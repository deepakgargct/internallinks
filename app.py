import streamlit as st
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from playwright.async_api import async_playwright

PRIMARY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

FALLBACK_HEADERS = {
    "User-Agent": "Googlebot/2.1 (+http://www.google.com/bot.html)"
}

async def fetch_html(session, url, use_fallback=False):
    headers = FALLBACK_HEADERS if use_fallback else PRIMARY_HEADERS
    try:
        async with session.get(url, headers=headers, timeout=15) as response:
            if response.status == 200:
                return await response.text()
    except Exception:
        return None
    return None

async def fetch_with_playwright(url):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, timeout=20000)
            content = await page.content()
            await browser.close()
            return content
    except Exception:
        return None

async def smart_fetch(session, url):
    html = await fetch_html(session, url, use_fallback=False)
    if not html:
        html = await fetch_html(session, url, use_fallback=True)
    if not html:
        html = await fetch_with_playwright(url)
    return html

def extract_clean_body(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
        tag.decompose()
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.decompose()
    for tag in soup.select("[class*=breadcrumb]"):
        tag.decompose()
    return soup.get_text(separator="\n")

def already_linked(html, target_url):
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True):
        if target_url.rstrip('/') in link['href'].rstrip('/'):
            return True
    return False

def find_link_opportunities(pages_content, target_content, keyword):
    if not pages_content:
        return []

    vectorizer = TfidfVectorizer().fit([target_content] + pages_content)
    tfidf_matrix = vectorizer.transform([target_content] + pages_content)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    opportunities = []
    for idx, score in enumerate(similarities):
        content = pages_content[idx]
        if keyword.lower() in content.lower():
            line_matches = [line for line in content.split("\n") if keyword.lower() in line.lower()]
            if line_matches:
                opportunities.append((idx, score, line_matches[0]))

    return sorted(opportunities, key=lambda x: x[1], reverse=True)

async def process_pages(page_urls, target_url, keyword):
    opportunities = []
    async with aiohttp.ClientSession() as session:
        target_html = await smart_fetch(session, target_url)
        if not target_html:
            st.error("❌ Could not fetch target page content.")
            return []

        target_text = extract_clean_body(target_html)

        tasks = [smart_fetch(session, url) for url in page_urls if url != target_url]
        results = await asyncio.gather(*tasks)

        cleaned_texts = []
        valid_urls = []
        for html, url in zip(results, page_urls):
            if html:
                if already_linked(html, target_url):
                    st.info(f"ℹ️ Skipping {url} — already links to target.")
                    continue

                cleaned = extract_clean_body(html)
                if cleaned.strip():
                    cleaned_texts.append(cleaned)
                    valid_urls.append(url)
                else:
                    st.info(f"⚠️ Skipping {url} — no useful content found.")
            else:
                st.warning(f"❌ Failed to fetch {url}")

        if not cleaned_texts:
            st.warning("No valid pages available for internal linking analysis.")
            return []

        raw_opportunities = find_link_opportunities(cleaned_texts, target_text, keyword)

        for idx, score, line in raw_opportunities:
            opportunities.append({
                "Page URL": valid_urls[idx],
                "Relevance Score": round(score, 2),
                "Suggested Anchor Line": line.strip()
            })

        return opportunities

# === Streamlit App ===
st.title("🔗 Internal Linking Opportunity Finder")
st.markdown("Upload a list of URLs and find which pages can internally link to your target page based on keyword and semantic relevance.")

uploaded_file = st.file_uploader("📄 Upload CSV/Excel with Page URLs", type=["csv", "xlsx"])
target_url = st.text_input("🔍 Target Page URL")
keyword = st.text_input("🏷️ Target Keyword")

if uploaded_file and target_url and keyword:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    url_column = [col for col in df.columns if "url" in col.lower()]
    if not url_column:
        st.error("❌ No URL column found in uploaded file.")
    else:
        page_urls = df[url_column[0]].dropna().unique().tolist()

        with st.spinner("🔄 Analyzing pages for linking opportunities..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(process_pages(page_urls, target_url, keyword))

        if results:
            df_out = pd.DataFrame(results)
            st.success(f"✅ Found {len(results)} internal linking opportunities.")
            st.dataframe(df_out)
            csv = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results as CSV", csv, "internal_link_opportunities.csv", "text/csv")
        else:
            st.warning("😕 No linking opportunities found.")
