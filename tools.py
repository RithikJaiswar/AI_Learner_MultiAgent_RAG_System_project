import requests
import os
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
load_dotenv()
from rich import print

# Tavily Tool

tavily = TavilyClient(api_key=os.getenv('TAVILY_API_KEY'))

@tool
def web_search(query:str)->str:
    '''Search the web for recent and reliable information on a topic. Return titles ,urls and snippets .'''
    results = tavily.search(query=query,max_results=5)
    out = []
    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL:{r['url']}\nSnippet:{r['content'][:300]}\n"
        )
    return "\n-----\n".join(out)


# BeautifulSoup Tool 

@tool
def web_scrap(url:str)->str:
    '''Scrape and return clean text content from a given URL for deep reading'''
    try:
        resp = requests.get(url,timeout=8,headers={'USER-Agent':'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text,'html.parser')
        for tag in soup(['script','style','nav','footer']):
            tag.decompose()
        return soup.get_text(separator=' ',strip=True)[:3000]
    except Exception as e:
        return f'Could not Scrape URL:{str(e)}'