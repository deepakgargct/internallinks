import streamlit as st
import pandas as pd
import requests
import nltk
from bs4 import BeautifulSoup

# Ensure the required NLTK data is available.
# This block checks if "punkt" exists and, if not, downloads it.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

def normalize_url(url):
    """Normalize URL strings for reliable comparison."""
    return url.strip().lower().rstrip("/")

def fetch_content(url):
    """
    Fetch page content using requests with a Google Bot User-Agent header.
    This helps bypass HTTP 403 errors from servers that allow Googlebot access.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            st.warning(f"Failed to fetch {url} (status: {response.status_code})")
            return None
    except Exception as e:
        st.error(f"Error fetching {url}: {e}")
        return None

def extract_paragraphs(html):
    """Extract text from all <p> tags in the provided HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
    return paragraphs

def tokenize_sentences(text):
    """Split provided text into a list of sentences using NLTK."""
    return nltk.sent_tokenize(text)

def find_internal_link_opportunities(page_url, page_text, target_list):
    """
    Look for internal linking opportunities by checking every sentence
    from the page's text for occurrences of target keywords.
    
    Args:
      page_url (str): The URL of the current page.
      page_text (str): HTML content of the page.
      target_list (list): List of dictionaries with keys 'target_url' and 'keyword'.
      
    Returns:
      list: A list of dictionaries where each dictionary represents a linking opportunity.
    """
    opportunities = []
    paragraphs = extract_paragraphs(page_text)
    for para in paragraphs:
        sentences = tokenize_sentences(para)
        for sentence in sentences:
            for target in target_list:
                target_url = target["target_url"]
                keyword = target["keyword"]
                # Avoid linking within the same page.
                if normalize_url(page_url) == normalize_url(target_url):
                    continue
                if keyword.lower() in sentence.lower():
                    opportunities.append({
                        "Page URL": page_url,
                        "Target URL": target_url,
                        "Keyword": keyword,
                        "Sentence": sentence
                    })
    return opportunities

def main():
    st.title("Internal Link Opportunity Finder")
    st.write(
        "Enter the page URLs to crawl along with the target URL and keyword pairs. "
        "The app will scan each page to identify sentences mentioning target keywords, "
        "suggesting potential internal linking opportunities."
    )

    # Text area for site URLs (one per line)
    sites_input = st.text_area(
        "Enter site URLs to crawl (one per line):",
        height=150,
        help="Example:\nhttps://example.com/page1\nhttps://example.com/page2"
    )

    # Text area for target URL and keyword pairs (comma-separated)
    targets_input = st.text_area(
        "Enter target URL and keyword pairs (one per line, format: target_url, keyword):",
        height=150,
        help="Example:\nhttps://example.com/about, team\nhttps://example.com/services, consulting"
    )

    if st.button("Find Internal Linking Opportunities"):
        if not sites_input.strip():
            st.error("Please provide at least one site URL.")
            return
        if not targets_input.strip():
            st.error("Please provide at least one target URL and keyword pair.")
            return

        # Process the site URLs input into a list
        site_urls = [line.strip() for line in sites_input.splitlines() if line.strip()]
        # Process the target pairs input into a list of dictionaries
        target_list = []
        for line in targets_input.splitlines():
            if line.strip():
                parts = line.split(",")
                if len(parts) < 2:
                    st.error(f"Invalid format for line: {line}. Expected: target_url, keyword")
                    return
                target_list.append({
                    "target_url": parts[0].strip(),
                    "keyword": ",".join(parts[1:]).strip()  # Handles keywords containing commas
                })
        
        all_opportunities = []
        for url in site_urls:
            st.write(f"Processing: {url}")
            content = fetch_content(url)
            if content:
                opps = find_internal_link_opportunities(url, content, target_list)
                if opps:
                    all_opportunities.extend(opps)
                else:
                    st.info(f"No internal linking opportunities found on {url}.")
            else:
                st.warning(f"Skipping {url} due to fetch error.")
        
        if all_opportunities:
            results_df = pd.DataFrame(all_opportunities)
            st.write("### Internal Linking Opportunities Found")
            st.dataframe(results_df)
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="internal_link_opportunities.csv",
                mime="text/csv"
            )
        else:
            st.info("No internal linking opportunities were identified based on the provided inputs.")

if __name__ == "__main__":
    main()
