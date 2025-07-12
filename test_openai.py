# ---------------------------------------------- Import all the necessary libraries -------------------------------------------------------------------

# !pip install --upgrade openai
# !pip install pinecone
# !pip install langchain
# !pip install langchain-community
# !pip install pandas
# !pip install dotenv
# !pip install numpy
# !pip install pydantic


from openai import OpenAI
import os
from pinecone import Pinecone, ServerlessSpec
from datetime import datetime

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.vectorstores import Chroma
from langchain.schema import Document

import pandas as pd 
import numpy as np

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import json
import re

# ----------------------------- Load the API Key ----------------------------------------------------------------------

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY') 
# os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')


client = OpenAI()



# ---------------------------------- First call to Open AI API for Website links ----------------------------------------------------------

prompt_websites = '''

Goal:
I want a list of the best News websites which shows the top engaging news from whole world. While returning, give 5 websites which we have given below and 5 websites from your side. Remember, the 5 websites that you will give needs to be different from those that we have provided you.

Our websites -

international-
https://www.theguardian.com/international
https://www.dw.com/en
https://www.cnn.com

indian-
https://indianexpress.com

china-
https://www.scmp.com


Return Format:
We will provide 5 websites and you have to add 5 more websites to the list .out of which 3 should be sites focused on global top news ,1 should be focused on news from india and 1 should be focused on china. In total, give me 10 websites


Warning:
Be careful to make sure that the name and website address is correct .and the websites are trustworthy to show true and unobstructed news.
as given above 5 sites should be included in the list and should follow the EXACT format as shown in example. 
Make sure in the end we have exactly 10 sites in the Final list. NO MORE NO LESS. 

Example: News Website:
international-
1.https://www.theguardian.com/international
2.https://www.dw.com/en
3.https://www.cnn.com
4.[Your site 1]
5.[Your site 2]
6.[Your site 3]

indian-
7.https://indianexpress.com
8.[Your site 4]


china-
9.https://www.scmp.com
10.[Your site 5]

'''

response = client.chat.completions.create(
    model="gpt-4o",  
    messages=[{"role": "user", "content": prompt_websites}],
    temperature=0.7
)

news_text = response.choices[0].message.content.strip()
print("📰 GPT Response:\n", news_text)




# ------------------------ Second call to OpenAI API for news Articles ----------------------------------------------------------------------------------


date= datetime.now().strftime("%Y-%m-%d")

class NewsItem(BaseModel):
    name: str = Field(..., alias="Name of news website")
    date: str  # You can also use datetime or a custom validator
    headline: str = Field(..., alias="News Headline")
    text: str = Field(..., alias="News Text")

class NewsSummary(BaseModel):
    items: List[NewsItem]

prompt_news = f'''

websites:
{news_text}

Goal:
Important - Main goal should be to provide the most engaging information so we can get the most user traffic on our websites.
For the links that I have provided above can you scrape the latest news present on all the webpages from the websites and give me the textual information in a structured format for the given date {date}?
Also summarize these news meaningfully in 3 sentences. i need further information so give read friendly tags like Location,Date,source


Note: I need the most engaging news from these websites. Please return the latest news (top 10), formatted **strictly as a JSON array**. The format must be:

Return Format:
The text that you scrape should be structured as follows:

json
[
  {{
    "Name of news website": ,
    "date": ,
    "News Headline": ,
    "News Text": 
  }},
]

Remember, you have to return top 10 news from all the websites. 7 should be from the links given above and 3 should be from your side which you think will garner the most engagement from the readers. 


Warning:
I don't want the unnecessary information such as google ads on the website or the text under Terms and Conditions footer information, etc.

Example:

json
[
  {{
    "Name of news website": "Bloomberg News",
    "date": "2025-06-08",
    "News Headline": "Tesla Shares drop to 53 %",
    "News Text": "Due to the fight between Trump and Elon Musk the company Tesla faces a major loss in share price. The drop is considered to be 53%."
  }},
]

'''

def call_gpt():
  response = client.responses.create(
      model="gpt-4o",
      tools=[{"type": "web_search_preview"}],
      input= prompt_news)
  return response


response= call_gpt()
# print(response.output_text)




# ------------------------------------------ Condition to check if the response contains news articles ---------------------------------------------------------------------

if "I cannot scrape" in response.output_text or "BeautifulSoup or Scrap" in response.output_text.lower():
    print("⚠️ GPT returned a non-informative scraping explanation.")
else:
    
    match = re.search(r'\[\s*\{.*?\}\s*\]', response.output_text, re.DOTALL)
    print(match)

    if match:
        json_string = match.group(0)
        try:
            news_data = json.loads(json_string)
            print("✅ Parsed JSON:")
            for news in news_data:
                print(json.dumps(news, indent=2))
        except json.JSONDecodeError as e:
            print("❌ Failed to parse JSON:", e)
    else:
        print("❌ JSON block not found in response.")



# ------------------------------------------------- Use news articles to create an image ---------------------------------------------------------------------------------

news_image = client.responses.create(
    model="gpt-4.1-mini",
    input=news_data[0]['News Text'],
    tools=[{"type": "image_generation"}],
)

news_image