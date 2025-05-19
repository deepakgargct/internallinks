import streamlit as st
import pandas as pd
import requests
import nltk
from bs4 import BeautifulSoup

# Download NLTK resources (only the first time the app runs)
nltk.download("punkt")

def normalize_url(url):
    """Normalize URLs for consistent comparisons."""
    return url.strip().lower().rstrip("/")

def fetch_content(url):
    """
    Fetch page content using requests and a custom User-Agent header.
    This helps bypass servers that block "non-browser" requests.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.85 Safari/537.36"
        )
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
    """Extract and return text from all <p> tags in the HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    paragraphs = [p.get_text().strip() for p in soup.find_all("p")]
    return paragraphs

def tokenize_sentences(text):
    """Divide provided text into a list of sentences."""
    return nltk.sent_tokenize(text)

def find_internal_link_opportunities(page_url, page_text, targets_df):
    """
    For each sentence in the page text, check whether it contains any target keyword.
    Skip pages that are the same as the target.
    """
    opportunities = []
    paragraphs = extract_paragraphs(page_text)
    for para in paragraphs:
        sentences = tokenize_sentences(para)
        for sentence in sentences:
            for _, row in targets_df.iterrows():
                target_url = row[0]
                keyword = row[1]
                # Avoid suggesting links within the same page
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
        "This tool identifies internal linking opportunities by scanning site pages and "
        "looking for sentences that mention target keywords. Upload the CSV files for the "
        "site URLs and target URL/keyword pairs to begin."
    )

    st.sidebar.header("Input Files")
    site_urls_file = st.sidebar.file_uploader("Upload site_urls.csv", type=["csv"])
    target_keywords_file = st.sidebar.file_uploader("Upload target_keywords.csv", type=["csv"])

    if site_urls_file and target_keywords_file:
        try:
            # Assume site_urls.csv contains one URL per row (no header)
            sites_df = pd.read_csv(site_urls_file, header=None, names=["site_url"])
            st.write("### Site URLs")
            st.dataframe(sites_df)
        except Exception as e:
            st.error(f"Error reading site_urls.csv: {e}")
            return

        try:
            # Assume target_keywords.csv contains two columns: target_url and keyword
            targets_df = pd.read_csv(target_keywords_file, header=None, names=["target_url", "keyword"])
            st.write("### Target URLs and Keywords")
            st.dataframe(targets_df)
        except Exception as e:
            st.error(f"Error reading target_keywords.csv: {e}")
            return

        if st.button("Find Internal Link Opportunities"):
            all_opportunities = []
            for idx, row in sites_df.iterrows():
                url = row["site_url"]
                st.write(f"Processing: {url}")
                content = fetch_content(url)
                if content:
                    opps = find_internal_link_opportunities(url, content, targets_df)
                    if opps:
                        all_opportunities.extend(opps)
                    else:
                        st.info(f"No opportunities found on {url}")
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
                st.info("No internal linking opportunities found!")
    else:
        st.info("Please upload both site_urls.csv and target_keywords.csv files.")

if __name__ == "__main__":
    main()
