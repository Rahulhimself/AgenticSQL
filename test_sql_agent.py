"""
Test script for direct SQL Sub-Agent connection and basic query execution.
Validates database credentials and zero-shot ReAct agent response generation.
"""

import os
import urllib.parse
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()

# Verify Google GenAI API key exists before connecting to Gemini
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing from your .env file!")

# Database configuration settings for local MS SQL Server
DB_USER = "langchain_agent"
raw_password = os.getenv("DB_PASSWORD")
if not raw_password:
    raise ValueError("DB_PASSWORD is missing from your .env file!")

DB_SERVER = "127.0.0.1" 
DB_NAME = "sql_practise"

# Construct ODBC connection string for SQL Server with SSL certificate trust
connection_string = (
    f"mssql+pyodbc://{DB_USER}:{raw_password}@{DB_SERVER}/{DB_NAME}"
    "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
)

# Initialize LangChain SQLDatabase connector
db = SQLDatabase.from_uri(connection_string)

# Initialize Gemini LLM with zero temperature for deterministic SQL generation
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    temperature=0
)

# Create zero-shot ReAct SQL agent with error parsing enabled
sql_sub_agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description",
    verbose=False,
    handle_parsing_errors=True
)

# Run a test query if executed directly from the terminal
if __name__ == "__main__":
    print("Testing SQL Sub-Agent connection...")
    test_prompt = "List all the tables present in this database and give me the datatypes for each column."
    response = sql_sub_agent.invoke({"input": test_prompt})
    print("\n--- Final Agent Response ---")
    print(response["output"])


