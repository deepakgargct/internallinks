import streamlit as st
import asyncio
import aiohttp
import re
from bs4 import BeautifulSoup
import pandas as pd
import difflib
from urllib.parse import urlparse, urljoin
import random

# --- Improved User-Agent list
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/124.0.0.0 Chrome/124.0.0.0 Safari/537.36",
]

# --- Async function to fetch page content
async def fetch_page_content(url):
    headers = {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    return None
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None

# --- Clean HTML to only body text (excluding header, footer, H1-H6, breadcrumbs)
def extract_main_content(html):
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove header, footer, nav, aside
    for tag in soup(["header", "footer", "nav", "aside"]):
        tag.decompose()

    # Remove H1-H6 headings
    for heading in soup.find_all(re.compile('^h[1-6]$')):
        heading.decompose()

    # Remove common breadcrumb classes/IDs
    for bc in soup.select(".breadcrumb, #breadcrumb, .breadcrumbs, #breadcrumbs"):
        bc.decompose()

    # Extract text
    text_blocks = []
    for p in soup.find_all(["p", "li", "span", "div"]):
        clean_text = p.get_text(strip=True)
        if clean_text and len(clean_text.split()) > 3:
            text_blocks.append(clean_text)
    return text_blocks

# --- Check if internal link already exists
def link_exists(text, keyword, target_url):
    keyword_lower = keyword.lower()
    if keyword_lower in text.lower():
        if target_url.lower() in text.lower():
            return True
    return False

# --- Find best match in text
def find_anchor_opportunity(text_blocks, keyword, target_url):
    opportunities = []
    for idx, block in enumerate(text_blocks):
        if link_exists(block, keyword, target_url):
            continue
        if keyword.lower() in block.lower():
            opportunities.append({
                "Suggested Anchor Text": keyword,
                "Text Line": block,
                "Text Line Number": idx + 1
            })
    return opportunities

# --- Streamlit App
st.set_page_config(page_title="🌐 Internal Linking Opportunity Finder", layout="wide")
st.title("🌐 Internal Linking Opportunity Finder")

uploaded_file = st.file_uploader("Upload CSV file with page URLs", type=["csv"])
target_url = st.text_input("Enter Target URL to Boost")
target_keyword = st.text_input("Enter Target Keyword for Internal Linking")

if uploaded_file and target_url and target_keyword:
    urls_df = pd.read_csv(uploaded_file)
    if "url" not in urls_df.columns:
        st.error("CSV must contain a 'url' column!")
    else:
        urls = urls_df['url'].dropna().tolist()
        results = []

        progress = st.progress(0)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for idx, url in enumerate(urls):
            if url.strip() == "" or url.strip() == target_url.strip():
                continue
            html = loop.run_until_complete(fetch_page_content(url))
            if html:
                text_blocks = extract_main_content(html)
                opportunities = find_anchor_opportunity(text_blocks, target_keyword, target_url)
                for opp in opportunities:
                    results.append({
                        "Source URL": url,
                        "Suggested Anchor Text": opp["Suggested Anchor Text"],
                        "Text Line": opp["Text Line"],
                        "Text Line Number": opp["Text Line Number"]
                    })
            progress.progress((idx + 1) / len(urls))

        if results:
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            st.download_button(
                label="📥 Download Internal Linking Opportunities CSV",
                data=df_results.to_csv(index=False).encode('utf-8'),
                file_name="internal_linking_opportunities.csv",
                mime="text/csv"
            )
        else:
            st.warning("No valid internal linking opportunities found.")

else:
    st.info("Please upload a CSV file and fill Target URL + Target Keyword.")

