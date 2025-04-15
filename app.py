import streamlit as st
import spacy
import subprocess
from bs4 import BeautifulSoup
import re
import pandas as pd

st.set_page_config(page_title="Internal Link Suggester", layout="wide")
st.title("🔗 Smart Internal Link Suggester")

# Attempt to load spaCy model, install if not available
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'nav', 'form']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True), soup

def extract_keywords(text):
    doc = nlp(text)
    keywords = set()

    for chunk in doc.noun_chunks:
        if len(chunk.text) > 2 and not chunk.text.lower() in ["this", "that", "those", "these"]:
            keywords.add(chunk.text.lower())

    return list(keywords)

def is_already_linked(keyword, soup):
    for a in soup.find_all('a', href=True):
        if keyword.lower() in a.get_text(strip=True).lower():
            return True
    return False

def find_insertion_snippet(text, keyword):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        if re.search(rf'\b{re.escape(keyword)}\b', sentence, re.IGNORECASE):
            return sentence.strip()
    return None

def generate_anchor_html(keyword, target_filename):
    href = target_filename.replace(" ", "%20")
    return f'<a href="{href}">{keyword}</a>'

uploaded_files = st.file_uploader("Upload HTML files", type=["html", "htm"], accept_multiple_files=True)

if uploaded_files:
    page_texts = {}
    page_soups = {}

    for file in uploaded_files:
        html_content = file.read().decode("utf-8", errors="ignore")
        text, soup = clean_html(html_content)
        page_texts[file.name] = text
        page_soups[file.name] = soup

    st.success("✅ Extracted and parsed all pages.")

    all_keywords = {}
    for filename, content in page_texts.items():
        keywords = extract_keywords(content)
        all_keywords[filename] = keywords

    st.subheader("🔍 Suggested Internal Linking Opportunities")
    all_matches = []

    for page, keywords in all_keywords.items():
        soup = page_soups[page]
        text = page_texts[page]

        for keyword in keywords:
            if is_already_linked(keyword, soup):
                continue

            for target_page, target_text in page_texts.items():
                if target_page == page:
                    continue

                if re.search(rf'\b{re.escape(keyword)}\b', target_text, re.IGNORECASE):
                    snippet = find_insertion_snippet(text, keyword)
                    if snippet:
                        all_matches.append({
                            "Anchor Text": keyword,
                            "Source Page": page,
                            "Target Page": target_page,
                            "Context Snippet": snippet,
                            "Suggested HTML": generate_anchor_html(keyword, target_page)
                        })

    if all_matches:
        df = pd.DataFrame(all_matches).drop_duplicates()
        st.dataframe(df)
        st.download_button("📥 Download Suggestions CSV", df.to_csv(index=False).encode("utf-8"), file_name="internal_link_suggestions.csv")
    else:
        st.info("No internal link opportunities found across these pages.")
