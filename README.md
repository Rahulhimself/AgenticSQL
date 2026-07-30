# AgenticSQL

A conversational AI backend that allows you to chat with a local Microsoft SQL Server database using natural language. Built with LangChain and Google's Gemini 3.6 Flash model. 

---

## What It Does
AgenticSQL acts as a bridge between normal human questions and complex database queries. Instead of writing manual T-SQL, you can ask the agent questions about your data (like "Give me a summary of the row counts in all tables"), and it will automatically inspect the database schema, write the correct SQL query, execute it, and return the answer in plain English.

## Key Features
* **Natural Language to SQL:** Uses a LangChain ReAct agent to autonomously query databases.
* **Powered by Gemini:** Integrates Google's current `gemini-3.6-flash` model for fast and accurate reasoning.
* **Local MS SQL Support:** Connects directly to local Microsoft SQL Server instances via PyODBC and SQLAlchemy.
* **Secure by Design:** Keeps database passwords and API keys safely out of the codebase using environment variables.

---

## Prerequisites
Before running this project, you will need:
* Python 3.x installed
* A local Microsoft SQL Server instance running with SQL Server Authentication enabled
* **ODBC Driver 17 for SQL Server** installed on your machine
* A Google AI Studio API key

## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/Rahulhimself/AgenticSQL.git](https://github.com/Rahulhimself/AgenticSQL.git)
cd AgenticSQL
```

**2. Install dependencies**

Install the required Python packages (it is recommended to use a Conda or virtual environment).
```bash
pip install langchain langchain-google-genai langchain-community sqlalchemy pyodbc python-dotenv
```
**3. Configure your environment variables**

Create a file named .env in the root directory of the project. Never upload this file to GitHub. Add your secure credentials to it:
```bash
GOOGLE_API_KEY="your_google_api_key_here"
DB_PASSWORD="your_database_password_here"
```
**4. Update database details**

Open test_sql_agent.py and ensure the following variables match your local SQL Server setup:
```bash
DB_USER (Your SQL Server login name)

DB_SERVER (Usually 127.0.0.1 for local instances)

DB_NAME (The database you want to query)
```
**Usage**

To test the agent and run a sample query against your database, execute the main script:
```bash
python test_sql_agent.py
```
The terminal will print the agent's step-by-step thought process as it connects to the database(Keep Verbose=True), inspects the tables, and retrieves your answer.




