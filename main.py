import sqlite3
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from prompts import *
import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)

# URL of the Manufacturing section of The Economic Times
# URL = "https://economictimes.indiatimes.com/industry/indl-goods/svs/engineering"


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def scrape_articles(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching the URL: {e}")
        return []

    soup = BeautifulSoup(response.content, "html.parser")
    articles = []
    
    for article in soup.find_all("div", class_="eachStory"):
        title = article.find("h3").get_text(strip=True)
        url = "https://economictimes.indiatimes.com" + article.find("a")["href"]
        publication_date = article.find("time").get_text(strip=True) if article.find("time") else "No Date"
        
        content = check_article_content(url)
        
        if content:
            articles.append({
                "title": title,
                "url": url,
                "publication_date": publication_date,
                "content": content
            })
        
        # Be respectful to the server by not sending requests too fast
        time.sleep(2)
    
    return articles

def check_article_content(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching article content: {e}")
        return None

    soup = BeautifulSoup(response.content, "html.parser")
    article_body = soup.find("div", {"class": "artText"})
    
    if article_body:
        content = article_body.get_text(strip=True)
    else:
        paragraphs = soup.find_all("p")
        content = " ".join([p.get_text(strip=True) for p in paragraphs])
    
    logging.info(f"Scraped content from {url}")
    return content


def get_db_text(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT content FROM articles")
    rows = cursor.fetchall()
    text = " ".join(row[0] for row in rows)
    conn.close()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template_1, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # **Warning:** This line allows dangerous deserialization
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat With Website")
    st.header("Chat with website using Gemini Pro")

    URL = st.text_input("Enter the website link", value="https://economictimes.indiatimes.com/industry/indl-goods/svs/engineering")
    if st.button("Load website"):
        with st.spinner("Loading..."):
                        # Scrape articles
            articles = scrape_articles(URL)

            # Convert to DataFrame for easier viewing and storage
            df = pd.DataFrame(articles)
            print(df.head())  # Preview the scraped data


            # Connect to SQLite database (or create it)
            conn = sqlite3.connect('articles.db')

            # Create a cursor object
            cur = conn.cursor()

            # Create the table to store articles
            cur.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                url TEXT,
                publication_date TEXT,
                content TEXT
            )
            ''')

            # Insert scraped articles into the table
            for _, row in df.iterrows():
                cur.execute('''
                INSERT INTO articles (title, url, publication_date, content)
                VALUES (?, ?, ?, ?)
                ''', (row['title'], row['url'], row['publication_date'], row['content']))

            # Commit the changes and close the connection
            conn.commit()
            cur.close()
            conn.close()

            print("Data saved to SQLite database.")

            # Query and print the saved data
            conn = sqlite3.connect('articles.db')
            cur = conn.cursor()

            cur.execute("SELECT * FROM articles")
            rows = cur.fetchall()

            for row in rows:
                print(row)

            cur.close()
            conn.close()

            db_path="articles.db"

            raw_text = get_db_text(db_path)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Database Loaded, Ask Your Queries!")

    user_question = st.text_input("Ask a Question from the Website")
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
