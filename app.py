import streamlit as st
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

st.set_page_config(page_title="Semantic Internal Linking Tool", layout="wide")
st.title("🔗 Internal Linking Suggestions (Semantic + Keyword Matching)")

# --- Step 1: Input Section ---
site_url = st.text_input("🌐 Root Domain (e.g., https://example.com)", "https://example.com")
target_url = st.text_input("🎯 Target Page URL", "https://example.com/contact-lens-hygiene-guide")
keyword = st.text_input("🔑 Target Keyword or Anchor Text", "contact lenses")

if st.button("🚀 Find Internal Linking Opportunities"):
    
    def get_internal_links(root_url):
        visited = set()
        to_visit = [root_url]
        internal_links = set()

        while to_visit:
            url = to_visit.pop()
            if url in visited or target_url in url:
                continue
            visited.add(url)

            try:
                response = requests.get(url, timeout=5)
                if not response.ok:
                    continue

                soup = BeautifulSoup(response.text, 'html.parser')
                for a in soup.find_all('a', href=True):
                    link = urljoin(url, a['href'])
                    if link.startswith(root_url) and '#' not in link:
                        parsed_link = link.split('?')[0].rstrip('/')
                        if parsed_link not in visited:
                            to_visit.append(parsed_link)
                            internal_links.add(parsed_link)
            except:
                continue

        return list(internal_links)

    def get_clean_text(url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'form']):
                tag.decompose()
            return soup.get_text(separator=" ", strip=True)
        except:
            return ""

    # Step 2: Crawl and Fetch Content
    st.info("🔍 Crawling website for internal URLs...")
    internal_urls = get_internal_links(site_url)
    st.success(f"Found {len(internal_urls)} internal URLs.")

    all_texts = {}
    for url in internal_urls:
        text = get_clean_text(url)
        if len(text) > 200:
            all_texts[url] = text

    st.info("📄 Fetching and cleaning content from internal pages...")

    # Step 3: Get target page content
    target_text = get_clean_text(target_url)
    if len(target_text) < 200:
        st.error("Target page content is too short or not accessible.")
        st.stop()

    # Step 4: Vectorize and Compare with TF-IDF
    corpus = [target_text] + list(all_texts.values())
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]

    # Step 5: Filter and Build Results
    threshold = 0.65
    matched_pages = []

    for i, (url, sim_score) in enumerate(zip(all_texts.keys(), similarities)):
        page_text = all_texts[url]
        if sim_score >= threshold and re.search(rf"\b{re.escape(keyword)}\b", page_text, re.IGNORECASE):
            snippet_match = re.search(rf"(.{0,50}{keyword}.{0,50})", page_text, re.IGNORECASE)
            snippet = snippet_match.group(0).strip() if snippet_match else "..."
            matched_pages.append({
                "Source Page": url,
                "Similarity Score": round(sim_score, 3),
                "Keyword Found In Snippet": snippet
            })

    # Step 6: Display
    if matched_pages:
        df = pd.DataFrame(sorted(matched_pages, key=lambda x: x["Similarity Score"], reverse=True))
        st.subheader("📈 Ranked Internal Linking Suggestions")
        st.dataframe(df)
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), file_name="internal_link_opportunities.csv")
    else:
        st.warning("No strong internal linking opportunities found. Try lowering the threshold or checking site accessibility.")
