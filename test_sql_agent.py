import os
import urllib.parse
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI

#load environment variables
load_dotenv()

#verify API key is detected
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY is missing from your .env file!")

#configure MS SQL Server Connection
DB_USER = "langchain_agent"
# Fetch the password securely from the .env file
raw_password = os.getenv("DB_PASSWORD")

#add a safety check just in case the .env file is missing the variable
if not raw_password:
    raise ValueError("DB_PASSWORD is missing from your .env file!")

#no 'tcp:' prefix Just the IP address
DB_SERVER = "127.0.0.1" 
DB_NAME = "sql_practise"

connection_string = (
    f"mssql+pyodbc://{DB_USER}:{raw_password}@{DB_SERVER}/{DB_NAME}"
    "?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
)

#initialize the SQLDatabase wrapper
db = SQLDatabase.from_uri(connection_string)

#initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash", 
    temperature=0
)

#create the SQL Agent
sql_sub_agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="zero-shot-react-description", # Changed from tool-calling
    verbose=False,
    handle_parsing_errors=True # Highly recommended for ReAct agents
)

#run a Test Query
if __name__ == "__main__":
    print("Testing SQL Sub-Agent connection...")
    
    test_prompt = "List all the tables present in this database and give me the datatypes for each column."
    
    response = sql_sub_agent.invoke({"input": test_prompt})
    
    print("\n--- Final Agent Response ---")
    print(response["output"])

