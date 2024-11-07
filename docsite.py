import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse

BASE_URL = "https://carbondesignsystem.com/" # Replace with your website's URL
visited_urls = set()
all_content = {}

def get_page_content(url):
  try:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    # Adjust this to target main content container
    content = soup.find("main") or soup.body # or a more specific tag if needed
    return content.get_text(strip=True) if content else ""
  except Exception as e:
    print(f"Failed to retrieve {url}: {e}")
    return ""

def crawl(url):
  if url in visited_urls:
    return
  print(f"Crawling {url}")
  visited_urls.add(url)
  page_content = get_page_content(url)
  all_content[url] = page_content

  # Find all internal links and crawl them
  response = requests.get(url)
  soup = BeautifulSoup(response.text, "html.parser")
  for link in soup.find_all("a", href=True):
    href = link["href"]
    full_url = urljoin(BASE_URL, href)
    # Ensure URL is within the same domain
    if urlparse(full_url).netloc == urlparse(BASE_URL).netloc:
      crawl(full_url)

# Start crawling from the base URL
crawl(BASE_URL)

# Save the combined content to a JSON file
with open("ibm_carbon_content_v1.json", "w", encoding="utf-8") as f:
  json.dump(all_content, f, indent=2, ensure_ascii=False)

print("Crawling complete. Content saved to ibm_carbon_content_v1.json")
