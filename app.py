import streamlit as st
import pandas as pd
import requests
import nltk
from bs4 import BeautifulSoup

# Download NLTK resources. This only needs to run once.
nltk.download("punkt")

def normalize_url(url):
    """Normalize URLs for comparison."""
    return url.strip().lower().rstrip("/")

@st.cache_data(show_spinner=False)
def fetch_content(url):
    """Fetch page content using requests."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            st.warning(f"Failed to fetch {url} (status: {response.status_code})")
            return None
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return None

def extract_paragraphs(html):
    """Extract text from <p> tags using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    return [p.get_text().strip() for p in soup.find_all("p")]

def tokenize_sentences(text):
    """Tokenize text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

def call_jina_api(sentence, keyword, timeout=10):
    """
    Call the Jina.ai API to perform semantic matching. 
    Replace the URL endpoint and add authentication details as required.
    
    For this example, we assume the API returns a JSON with a 'score' field.
    """
    api_endpoint = "https://api.jina.ai/semantic-search"  # <-- Replace with your actual endpoint
    payload = {
        "sentence": sentence,
        "keyword": keyword
    }
    try:
        response = requests.post(api_endpoint, json=payload, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            # Example: API might return {"score": 0.78}
            return data.get("score", 0)
        else:
            return 0
    except Exception as e:
        st.error(f"Error communicating with Jina API: {e}")
        return 0

def find_opportunities_with_jina(page_url, html_content, targets, similarity_threshold=0.5):
    """
    For each sentence extracted from the page, use the Jina.ai API to determine
    whether the sentence semantically matches the target keyword.
    """
    opportunities = []
    paragraphs = extract_paragraphs(html_content)
    for para in paragraphs:
        sentences = tokenize_sentences(para)
        for sentence in sentences:
            for target_item in targets:
                target_url = target_item["target_url"]
                keyword = target_item["keyword"]
                # Avoid linking a page to itself.
                if normalize_url(page_url) == normalize_url(target_url):
                    continue
                # Call the Jina.ai API to get a semantic similarity score.
                score = call_jina_api(sentence, keyword)
                if score >= similarity_threshold:
                    opportunities.append({
                        "Page URL": page_url,
                        "Target URL": target_url,
                        "Keyword": keyword,
                        "Sentence": sentence,
                        "Similarity Score": score
                    })
    return opportunities

def main():
    st.title("Internal Links Opportunity Finder with Jina.ai Integration")
    st.write(
        "Instead of uploading CSV files, simply enter the site URLs to crawl and the "
        "targeted page and keyword pairs below. The app will fetch the content from the sites "
        "and use the Jina.ai API to detect linking opportunities based on semantic similarity."
    )
    
    # Sidebar inputs for site URLs and target page/keyword pairs.
    st.header("Input Data")
    
    sites_input = st.text_area(
        "Enter site URLs to crawl (one per line):", 
        height=150, 
        help="Example:\nhttps://example.com/page1\nhttps://example.com/page2"
    )

    targets_input = st.text_area(
        "Enter targeted pages with associated keyword (format: target_url, keyword, one per line):", 
        height=150, 
        help="Example:\nhttps://example.com/about, team\nhttps://example.com/services, consulting"
    )

    if st.button("Find Internal Linking Opportunities"):
        # Validate input.
        if not sites_input.strip():
            st.error("Please provide at least one site URL to crawl.")
            return
        if not targets_input.strip():
            st.error("Please provide at least one target URL and keyword pair.")
            return
            
        # Process site URLs (one per line).
        site_urls = [line.strip() for line in sites_input.splitlines() if line.strip()]
        
        # Process target entries.
        targets = []
        for line in targets_input.splitlines():
            if line.strip():
                parts = line.split(',')
                if len(parts) < 2:
                    st.error(f"Invalid format in the line: {line}. Expected format: target_url, keyword")
                    return
                target_url = parts[0].strip()
                keyword = ','.join(parts[1:]).strip()  # In case the keyword contains commas.
                targets.append({"target_url": target_url, "keyword": keyword})
        
        all_opportunities = []
        for url in site_urls:
            st.write(f"Processing: {url}")
            content = fetch_content(url)
            if content:
                opps = find_opportunities_with_jina(url, content, targets)
                if opps:
                    all_opportunities.extend(opps)
                else:
                    st.info(f"No internal linking opportunities found on {url}.")
            else:
                st.warning(f"Skipping {url} because content could not be fetched.")
        
        if all_opportunities:
            results_df = pd.DataFrame(all_opportunities)
            st.write("### Internal Linking Opportunities Found")
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="internal_link_opportunities.csv",
                mime="text/csv"
            )
        else:
            st.info("No internal linking opportunities found for the provided inputs.")

if __name__ == "__main__":
    main()
