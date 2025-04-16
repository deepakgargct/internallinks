import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ----------------- Utility Functions -------------------

def get_clean_text(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'form']):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return ""

def get_internal_links(site_url):
    visited = set()
    to_visit = [site_url]
    internal_links = set()
    domain = urlparse(site_url).netloc

    while to_visit:
        current_url = to_visit.pop()
        if current_url in visited or urlparse(current_url).netloc != domain:
            continue
        visited.add(current_url)
        try:
            response = requests.get(current_url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0"
            })
            if response.status_code != 200:
                continue
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = urljoin(current_url, link['href'])
                if domain in urlparse(href).netloc and href not in visited:
                    internal_links.add(href)
                    to_visit.append(href)
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
            continue
    return list(internal_links)

def find_internal_link_opportunities(site_url, target_url, keyword, threshold=0.65):
    pages = get_internal_links(site_url)
    pages = [url for url in pages if url != target_url]

    target_text = get_clean_text(target_url)
    if not target_text:
        st.error("Failed to fetch content from the target URL.")
        return []

    candidates = []
    texts = []

    for page in pages:
        text = get_clean_text(page)
        if keyword.lower() in text.lower() and len(text) > 100:
            candidates.append(page)
            texts.append(text)

    if not texts:
        return []

    all_texts = [target_text] + texts
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    ranked = [(candidates[i], similarities[i]) for i in range(len(candidates))]
    ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
    return [(url, score) for url, score in ranked if score > threshold]

# ------------------ Streamlit App -----------------------

st.title("🔗 Internal Linking Opportunities Finder")

site_url = st.text_input("Website URL (e.g. https://example.com)")
target_url = st.text_input("Target Page URL (e.g. https://example.com/contact-lens-hygiene-guide)")
keyword = st.text_input("Target Keyword (e.g. contact lenses)")

if st.button("Find Internal Links"):
    if not site_url or not target_url or not keyword:
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Crawling site and analyzing pages..."):
            links = find_internal_link_opportunities(site_url, target_url, keyword)
        
        if links:
            st.success(f"Found {len(links)} internal linking opportunities.")
            for url, score in links:
                st.markdown(f"- [{url}]({url}) — Similarity Score: `{score:.2f}`")
        else:
            st.info("No relevant internal linking opportunities found.")
