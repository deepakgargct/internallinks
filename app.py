import httpx
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import re
import asyncio
import aiohttp

# Fetch page content with headers
async def fetch_page_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=20) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    return f"Failed to fetch {url}: {response.status}"
        except Exception as e:
            return f"Error fetching {url}: {e}"

# Extract internal links from a page
def extract_internal_links(content, site_url):
    soup = BeautifulSoup(content, 'html.parser')
    links = []
    
    # Extract all links that are within the site (i.e., internal links)
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.startswith('/') or href.startswith(site_url):
            if site_url not in href:  # Fix any relative links or malformed links
                href = site_url + href
            links.append(href)
    return set(links)

# Extract main content excluding H1-H6, header, footer
def extract_main_content(content):
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove unwanted elements
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'footer']):
        tag.decompose()

    # Extract the remaining text
    return ' '.join([p.text for p in soup.find_all('p')])

# Find internal link opportunities based on semantic similarity (simple keyword match for now)
def find_link_opportunities(pages, target_url, target_keyword):
    opportunities = []
    
    for url, content in pages.items():
        if target_url in url:  # Skip the target URL itself
            continue
        if target_keyword.lower() in content.lower():
            opportunities.append(url)
    
    return opportunities

# Streamlit UI
def main():
    st.title("Internal Linking Opportunity Finder")

    # File Upload
    uploaded_file = st.file_uploader("Upload CSV or Excel file with page URLs", type=["csv", "xlsx"])

    if uploaded_file:
        # Read the uploaded file into a pandas dataframe
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("URLs in the uploaded file:")
        st.write(df)

        # User inputs for target URL and keyword
        target_url = st.text_input("Enter the target URL", "")
        target_keyword = st.text_input("Enter the targeted keyword", "")

        if st.button("Find Internal Linking Opportunities"):
            if target_url and target_keyword:
                pages = {}
                # Fetch content for each URL
                for url in df['URL']:
                    st.write(f"Processing {url}...")
                    content = asyncio.run(fetch_page_content(url))
                    if "Failed" not in content:  # If the content was fetched successfully
                        main_content = extract_main_content(content)
                        pages[url] = main_content
                    else:
                        st.write(content)
                
                # Find internal linking opportunities
                opportunities = find_link_opportunities(pages, target_url, target_keyword)
                
                st.write("Found the following internal linking opportunities:")
                st.write(opportunities)
                
                # Download the list of opportunities as a CSV
                opportunity_df = pd.DataFrame(opportunities, columns=["Internal Link Opportunities"])
                st.download_button(
                    label="Download Opportunities",
                    data=opportunity_df.to_csv(index=False).encode('utf-8'),
                    file_name="internal_link_opportunities.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
