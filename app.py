import streamlit as st
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re
from collections import defaultdict

# Function to fetch the content of the page
async def fetch_page_content(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                html = await response.text()
                return html
            else:
                return None

# Function to extract internal links
def extract_internal_links(base_url, html):
    soup = BeautifulSoup(html, "html.parser")
    links = set()

    for anchor in soup.find_all('a', href=True):
        href = anchor['href']
        # Ensure the link is internal
        if base_url in href:
            full_url = urljoin(base_url, href)
            links.add(full_url)

    return links

# Function to extract body content and exclude headers and footers
def extract_body_content(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Remove header, footer, and h1-h6 tags
    for tag in soup.find_all(['header', 'footer', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        tag.decompose()

    # Extract body content
    body = soup.find('body')
    return body.get_text(separator=' ', strip=True) if body else ""

# Function to find the suggested anchor text lines
def find_suggested_anchor_lines(content, keyword):
    lines = content.split('.')
    relevant_lines = []
    for line in lines:
        if keyword.lower() in line.lower():
            relevant_lines.append(line.strip())
    return relevant_lines

# Function to process CSV and find internal linking opportunities
def process_file(file, target_url, seed_keyword):
    # Read the CSV or Excel file into a DataFrame
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    else:
        return "Invalid file format. Please upload a CSV or Excel file."
    
    # Clean up and ensure 'URL' column exists
    df = df.dropna(subset=['URL'])
    base_url = target_url.rstrip('/').rsplit('/', 1)[0]  # Get base URL without trailing slash
    
    opportunities = []
    
    # Loop through each URL in the file
    for index, row in df.iterrows():
        page_url = row['URL']
        html = asyncio.run(fetch_page_content(page_url))

        if html:
            page_content = extract_body_content(html)
            internal_links = extract_internal_links(base_url, html)

            # Find lines that mention the seed keyword
            suggested_lines = find_suggested_anchor_lines(page_content, seed_keyword)

            # Check if the seed keyword is present in the page content and extract relevant anchor text suggestions
            for line in suggested_lines:
                if line:
                    opportunities.append({
                        'Page URL': page_url,
                        'Suggested Anchor Text': seed_keyword,
                        'Anchor Text Line': line,
                        'Internal Links Found': internal_links
                    })
        else:
            opportunities.append({
                'Page URL': page_url,
                'Suggested Anchor Text': seed_keyword,
                'Anchor Text Line': 'Content not found or page inaccessible.',
                'Internal Links Found': []
            })

    return opportunities

# Streamlit interface
st.title("Internal Linking Opportunity Finder")

# File upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file containing URLs", type=["csv", "xlsx"])

if uploaded_file:
    # Input fields for Target URL and Seed Keyword
    target_url = st.text_input("Enter Target URL (the page you want to create internal links for)")
    seed_keyword = st.text_input("Enter Seed Keyword or Anchor Text")

    if target_url and seed_keyword:
        # Process the file and find internal linking opportunities
        opportunities = process_file(uploaded_file, target_url, seed_keyword)

        if opportunities:
            # Display the internal linking opportunities
            opportunities_df = pd.DataFrame(opportunities)
            st.write(opportunities_df)

            # Provide option to download the output as CSV
            st.download_button(
                label="Download Internal Linking Opportunities",
                data=opportunities_df.to_csv(index=False),
                file_name="internal_linking_opportunities.csv",
                mime="text/csv"
            )
        else:
            st.write("No internal linking opportunities found.")
