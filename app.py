import streamlit as st
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
}

# Initialize Sentence-Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

async def fetch_html(session, url):
    try:
        async with session.get(url, headers=HEADERS, timeout=15) as response:
            if response.status == 200:
                return await response.text()
            else:
                return None
    except Exception as e:
        st.warning(f"Error fetching {url}: {e}")
        return None

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
        return []  # Return an empty list if there's nothing to compare

    # Encode target content and pages using sentence embeddings
    target_embedding = model.encode([target_content])
    page_embeddings = model.encode(pages_content)

    similarities = cosine_similarity(target_embedding, page_embeddings).flatten()

    opportunities = []
    for idx, score in enumerate(similarities):
        if keyword.lower() in pages_content[idx].lower():
            for line in pages_content[idx].split(". "):
                if keyword.lower() in line.lower():
                    opportunities.append((idx, score, line.strip()))
                    break
    return sorted(opportunities, key=lambda x: x[1], reverse=True)

async def process_pages(page_urls, target_url, keyword):
    opportunities = []
    async with aiohttp.ClientSession() as session:
        target_html = await fetch_html(session, target_url)
        if not target_html:
            st.error("Target page content not found.")
            return []

        target_text = extract_clean_body(target_html)

        tasks = [fetch_html(session, url) for url in page_urls if url != target_url]
        results = await asyncio.gather(*tasks)

        cleaned_texts = []
        valid_urls = []
        for html, url in zip(results, page_urls):
            if html and not already_linked(html, target_url):
                cleaned = extract_clean_body(html)
                if cleaned.strip():
                    cleaned_texts.append(cleaned)
                    valid_urls.append(url)

        if not cleaned_texts:
            return []  # Return an empty list if no valid pages were found

        raw_opportunities = find_link_opportunities(cleaned_texts, target_text, keyword)

        for idx, score, line in raw_opportunities:
            opportunities.append({
                "Page URL": valid_urls[idx],
                "Relevance Score": round(score, 2),
                "Suggested Anchor Line": line.strip()
            })

        return opportunities

st.title("Internal Linking Opportunity Finder")
st.markdown("This tool finds content sections from other pages that can internally link to your target page.")

uploaded_file = st.file_uploader("Upload CSV/Excel with Page URLs", type=["csv", "xlsx"])
target_url = st.text_input("Target Page URL")
keyword = st.text_input("Target Keyword")

if uploaded_file and target_url and keyword:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    url_column = [col for col in df.columns if "url" in col.lower()]
    if not url_column:
        st.error("Could not find a column with URLs.")
    else:
        page_urls = df[url_column[0]].dropna().unique().tolist()

        with st.spinner("Scanning pages and analyzing opportunities..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(process_pages(page_urls, target_url, keyword))

        if results:
            df_out = pd.DataFrame(results)
            st.success(f"Found {len(results)} linking opportunities.")
            st.dataframe(df_out)
            csv = df_out.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results as CSV", csv, "internal_link_opportunities.csv", "text/csv")
        else:
            st.warning("No linking opportunities found.")
