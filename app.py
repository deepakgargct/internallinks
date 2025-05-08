import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

st.title("🔗 Internal Link Recommender for Local Business Sites")

website_url = st.text_input("Enter your website URL (e.g., https://example.com)")
target_url = st.text_input("Enter your target page URL (e.g., https://example.com/services/example)")

submitted = st.button("Analyze Website")

def get_main_content(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in ['header', 'footer', 'nav', 'aside', 'script', 'style']:
        for element in soup.find_all(tag):
            element.decompose()
    body = soup.find('body')
    return str(body) if body else str(soup)

def get_internal_links(base_url):
    visited = set()
    to_visit = [base_url]
    internal_links = []

    while to_visit:
        url = to_visit.pop()
        if url in visited or not url.startswith(base_url):
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=10)
            cleaned_html = get_main_content(response.text)
            internal_links.append((url, cleaned_html))

            soup = BeautifulSoup(response.text, "html.parser")
            for a_tag in soup.find_all("a", href=True):
                href = urljoin(url, a_tag['href'])
                if base_url in href and href not in visited:
                    to_visit.append(href)

        except Exception as e:
            st.warning(f"Error fetching {url}: {e}")

    return internal_links

def extract_sentences(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ").strip()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 40]

def already_links_to(content, target):
    return target in content

def suggest_anchor_text(sentence, keyword):
    doc = nlp(sentence)
    for chunk in doc.noun_chunks:
        if keyword.lower() in chunk.text.lower():
            return chunk.text
    return keyword

def find_recommendations(pages, target_url):
    recommendations = []
    target_keyword = urlparse(target_url).path.strip("/").split("/")[-1].replace("-", " ")

    vectorizer = TfidfVectorizer().fit([target_keyword])

    for page_url, html_content in pages:
        if already_links_to(html_content, target_url):
            continue

        sentences = extract_sentences(html_content)
        if not sentences:
            continue

        scores = cosine_similarity(vectorizer.transform(sentences), vectorizer.transform([target_keyword])).flatten()
        ranked_sentences = sorted(zip(sentences, scores), key=lambda x: -x[1])

        for sent, score in ranked_sentences[:2]:  # Top 2 per page
            anchor_text = suggest_anchor_text(sent, target_keyword)
            new_sentence = sent.replace(anchor_text, f"[{anchor_text}]({target_url})")
            recommendations.append({
                "Source Page URL": page_url,
                "Anchor Text": anchor_text,
                "Sentence with Link Placement": new_sentence,
                "Suggested Insertion Point": f"After: \"{sent[:80]}...\""
            })

        if len(recommendations) >= 20:
            break

    return pd.DataFrame(recommendations)

if submitted and website_url and target_url:
    st.info("Crawling and analyzing internal pages. Please wait...")
    internal_pages = get_internal_links(website_url)
    st.success(f"Found {len(internal_pages)} internal pages.")
    df = find_recommendations(internal_pages, target_url)

    if df.empty:
        st.warning("No suitable internal link opportunities found.")
    else:
        st.dataframe(df)
        st.download_button("Download Recommendations as CSV", df.to_csv(index=False), "internal_links.csv")
