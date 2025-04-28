import streamlit as st
import pandas as pd
import asyncio
import httpx
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(page_title="Internal Linking Tool", layout="wide")

st.title("🔗 Smart Internal Linking Opportunities Finder")

# --- Async content fetcher
async def fetch_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    async with httpx.AsyncClient(timeout=20, follow_redirects=True, headers=headers) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Error {response.status_code} fetching {url}")
                return None
        except Exception as e:
            print(f"Exception fetching {url}: {e}")
            return None

# --- Clean content (remove headers, footers, h1-h6, nav, breadcrumbs)
def clean_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove header, footer, nav, aside, breadcrumbs
    for tag in soup.find_all(['header', 'footer', 'nav', 'aside', 'breadcrumb']):
        tag.decompose()

    # Remove H1-H6 tags
    for header_tag in soup.find_all(re.compile('^h[1-6]$')):
        header_tag.decompose()

    # Get remaining text
    body_text = soup.get_text(separator=" ", strip=True)
    return body_text

# --- Check if link already exists
def link_exists(html_content, target_url):
    return target_url in html_content

# --- Find anchor text placement
def find_anchor_line(body_text, seed_keyword):
    lines = body_text.split('. ')
    for line in lines:
        if seed_keyword.lower() in line.lower():
            return line.strip()
    return None

# --- Streamlit App Logic
uploaded_file = st.file_uploader("📄 Upload CSV or Excel file containing Page URLs:", type=["csv", "xlsx"])

target_url = st.text_input("🔗 Enter Target URL (where you want links pointing to):")
seed_keyword = st.text_input("📝 Enter Seed Keyword / Anchor Text:")

if uploaded_file and target_url and seed_keyword:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Try to find URL column
        url_column = None
        for col in df.columns:
            if 'url' in col.lower():
                url_column = col
                break

        if not url_column:
            st.error("Could not find URL column in your file. Please make sure there is a column with 'url' in the name.")
        else:
            urls = df[url_column].dropna().tolist()

            # Exclude target_url itself
            urls = [url for url in urls if url != target_url]

            st.info(f"Found {len(urls)} pages to analyze... crawling now 🚀")

            async def process_urls(urls, target_url, seed_keyword):
                target_html = await fetch_content(target_url)
                if not target_html:
                    st.error(f"Failed to fetch target URL content: {target_url}")
                    return []

                target_text = clean_html(target_html)

                results = []
                tasks = []
                for url in urls:
                    tasks.append(fetch_content(url))

                pages_html = await asyncio.gather(*tasks)

                for i, page_html in enumerate(pages_html):
                    if not page_html:
                        continue

                    if link_exists(page_html, target_url):
                        # Link already exists, skip it
                        continue

                    page_text = clean_html(page_html)

                    # Semantic similarity check
                    vectorizer = TfidfVectorizer().fit_transform([target_text, page_text])
                    vectors = vectorizer.toarray()
                    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

                    if cosine_sim >= 0.4:  # threshold
                        anchor_line = find_anchor_line(page_text, seed_keyword)
                        if anchor_line:
                            results.append({
                                "Source Page": urls[i],
                                "Suggested Anchor Line": anchor_line,
                                "Similarity Score": round(cosine_sim, 3)
                            })

                return results

            results = asyncio.run(process_urls(urls, target_url, seed_keyword))

            if results:
                results_df = pd.DataFrame(results)
                st.success(f"✅ Found {len(results)} internal linking opportunities!")
                st.dataframe(results_df)

                # Download button
                st.download_button(
                    label="📥 Download Opportunities as CSV",
                    data=results_df.to_csv(index=False),
                    file_name="internal_linking_opportunities.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No valid pages found for internal linking.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
