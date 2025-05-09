import streamlit as st
import requests
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser
from readability import Document
from bs4 import BeautifulSoup
import re
from rake_nltk import Rake
import pandas as pd
import time
from io import BytesIO

st.title("SEO Internal Linking Advisor")

# User inputs
site_url = st.text_input("Enter the website URL to analyze (e.g. https://example.com):")
target_url = st.text_input("Enter the target page URL (the page to receive internal links):")

if st.button("Analyze"):
    if not site_url or not target_url:
        st.error("Please provide both the website and target page URLs.")
    else:
        # Normalize base site URL
        parsed_site = urlparse(site_url)
        base_domain = f"{parsed_site.scheme}://{parsed_site.netloc}"
        
        # Setup robots.txt parsing
        robots_url = urljoin(base_domain, "/robots.txt")
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except:
            rp = None
        
        # Get crawl-delay if specified
        delay = 0
        if rp:
            crawl_delay = rp.crawl_delay("*")
            if crawl_delay:
                delay = crawl_delay
        
        # Prepare to crawl
        to_crawl = [site_url]
        visited = set()
        candidates = []

        # Crawl internal pages
        while to_crawl:
            url = to_crawl.pop(0)
            if url in visited:
                continue
            visited.add(url)
            # Respect robots.txt rules
            if rp and not rp.can_fetch("*", url):
                continue
            try:
                response = requests.get(url, timeout=10)
            except requests.RequestException:
                continue
            time.sleep(delay or 0)  # Polite crawling
            
            if response.status_code != 200:
                continue
            
            html = response.text
            # Parse and find new internal links
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a['href']
                href = urljoin(url, href)
                parsed_href = urlparse(href)
                if parsed_href.netloc == parsed_site.netloc:
                    # Remove fragment and query
                    clean = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
                    if clean not in visited and clean not in to_crawl:
                        to_crawl.append(clean)
            
            # Extract main content using readability
            doc = Document(html)
            main_html = doc.summary()
            main_text = BeautifulSoup(main_html, "lxml").get_text(separator=" ")
            
            # Check if page already links to target
            if soup.find("a", href=target_url):
                continue
            
            candidates.append((url, main_html, main_text))
        
        # If no candidates found
        if not candidates:
            st.write("No suitable pages found or all pages already link to the target.")
        else:
            # Extract keywords from target page using RAKE
            try:
                target_resp = requests.get(target_url)
                target_doc = Document(target_resp.text)
                target_main = BeautifulSoup(target_doc.summary(), "lxml").get_text(separator=" ")
            except:
                target_main = ""
            rake = Rake()
            rake.extract_keywords_from_text(target_main)
            key_phrases = rake.get_ranked_phrases()
            
            # Filter and prioritize keywords
            key_phrases = [phrase for phrase in key_phrases if phrase and len(phrase.split()) <= 4]
            
            results = []
            # Search for link opportunities
            for page_url, page_html, page_text in candidates:
                soup_page = BeautifulSoup(page_html, "lxml")
                text = soup_page.get_text(separator=" ")
                sentences = re.split(r'(?<=[.!?]) +', text)
                
                for sentence in sentences:
                    sent_lower = sentence.lower()
                    for phrase in key_phrases:
                        if phrase.lower() in sent_lower:
                            # Proposed anchor text
                            anchor = phrase
                            # Determine paragraph position
                            paragraphs = [p.get_text() for p in soup_page.find_all(["p", "li"])]
                            insert_pos = None
                            for idx, p_text in enumerate(paragraphs, 1):
                                if sentence.strip() in p_text:
                                    insert_pos = f"Paragraph {idx}"
                                    break
                            if not insert_pos:
                                insert_pos = "Content body"
                            results.append({
                                "Source Page URL": page_url,
                                "Anchor Text": anchor,
                                "Sentence with Link Placement": sentence.strip(),
                                "Suggested Insertion Point": insert_pos
                            })
                            break  # use first matching phrase
                    if len(results) >= 20:
                        break
                if len(results) >= 20:
                    break
            
            # Prepare DataFrame and output
            df = pd.DataFrame(results[:20])
            df = df[["Source Page URL", "Anchor Text", "Sentence with Link Placement", "Suggested Insertion Point"]]
            
            towrite = BytesIO()
            df.to_excel(towrite, index=False, sheet_name="Internal Links")
            towrite.seek(0)
            st.download_button(
                label="Download Internal Link Suggestions",
                data=towrite,
                file_name="internal_link_suggestions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
