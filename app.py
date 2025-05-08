import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Load NLP and semantic models
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-mpnet-base-v2")

st.title("🔗 Internal Linking Recommender (TF-IDF + BERT)")

# Input fields
website_url = st.text_input("Enter your website URL (e.g., https://example.com)")
target_url = st.text_input("Enter your target page URL (e.g., https://example.com/services/example)")
target_keyword = st.text_input("Enter your target keyword (e.g., roofing services)")

submitted = st.button("Analyze Website")

# Utility: clean and extract main content
def get_main_content(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in ['header', 'footer', 'nav', 'aside', 'script', 'style']:
        for element in soup.find_all(tag):
            element.decompose()
    return soup

# Crawl internal links
def get_internal_links(base_url):
    visited = set()
    to_visit = [base_url]
    internal_pages = []

    while to_visit:
        url = to_visit.pop()
        if url in visited or not url.startswith(base_url):
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = get_main_content(response.text)
                internal_pages.append((url, soup))

                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    if link.startswith(base_url) and link not in visited:
                        to_visit.append(link)
        except:
            continue

    return internal_pages

# Extract clean sentences from HTML soup
def extract_sentences(soup):
    text_blocks = soup.get_text(separator=" ").strip()
    doc = nlp(text_blocks)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 40]

# Check if the sentence already links to the target URL
def already_links_to(soup, target_url):
    for a in soup.find_all("a", href=True):
        if target_url in a["href"]:
            return True
    return False

# Find best anchor text inside the sentence
def find_anchor_text(sentence, keyword):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        if keyword.lower() in chunk.text.lower():
            return chunk.text
    return keyword  # fallback

# Recommend internal links based on TF-IDF and BERT
def recommend_links(pages, target_url, keyword):
    candidates = []
    sentences = []

    for url, soup in pages:
        if already_links_to(soup, target_url):
            continue
        sents = extract_sentences(soup)
        for idx, sent in enumerate(sents):
            if keyword.lower() in sent.lower():  # prefilter
                sentences.append((url, idx, sent))

    if not sentences:
        return pd.DataFrame()

    # TF-IDF Scoring
    corpus = [s[2] for s in sentences]
    tfidf_vec = TfidfVectorizer().fit(corpus + [keyword])
    tfidf_matrix = tfidf_vec.transform(corpus)
    tfidf_scores = cosine_similarity(tfidf_vec.transform([keyword]), tfidf_matrix).flatten()

    # BERT Scoring
    embeddings = bert_model.encode(corpus + [keyword], convert_to_tensor=True)
    bert_scores = util.cos_sim(embeddings[-1], embeddings[:-1])[0].cpu().numpy()

    # Combine and rank
    results = []
    for (url, idx, sent), tfidf, bert in zip(sentences, tfidf_scores, bert_scores):
        score = 0.4 * tfidf + 0.6 * bert  # weighted relevance
        anchor_text = find_anchor_text(sent, keyword)
        results.append({
            "Source Page URL": url,
            "Anchor Text": anchor_text,
            "Sentence with Link Placement": sent.replace(anchor_text, f"[{anchor_text}]({target_url})"),
            "Suggested Insertion Point": f"After sentence #{idx + 1}",
            "Relevance Score": round(score, 4)
        })

    top_results = sorted(results, key=lambda x: -x["Relevance Score"])[:20]
    return pd.DataFrame(top_results)

# Run analysis
if submitted and website_url and target_url and target_keyword:
    st.info("🔍 Crawling website and analyzing link opportunities...")
    internal_pages = get_internal_links(website_url)
    st.success(f"✅ Found {len(internal_pages)} internal pages.")
    df = recommend_links(internal_pages, target_url, target_keyword)

    if df.empty:
        st.warning("No strong internal linking opportunities found.")
    else:
        st.dataframe(df.drop(columns=["Relevance Score"]))
        st.download_button("📥 Download as CSV", df.to_csv(index=False), "internal_links.csv")
