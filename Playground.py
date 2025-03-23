from phi.agent  import Agent
# Ensure the module is installed or correct the import path
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools 
from phi.tools.duckduckgo import DuckDuckGo 
import  openai
from  dotenv import load_dotenv

import os
import phi
from phi.playground import Playground, serve_playground_app

load_dotenv()

phi.api = os.getenv("PHI_API_KEY")


# web Search  Agent:
web_search_agent = Agent(
    name = "web search agent",
    role = "Search the web for the Informataion",
    model = Groq(id = "llama3-70b-8192-tool-use-preview"),
    tools = [DuckDuckGo()],
    instruction = ["Always included sources"],
    show_tools_Calls = True,
    markdown = True,   
)

# Financial AI Agent:
Finance_agent  = Agent(
    name = "Finance AI Agent",
    model = Groq(id="llama3-70b-8192-tool-use-preview"),
    tools  = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news= True  ),
    ],
    instruction = ["Use Tables  to display the Data"],
    show_tools_calls= True,
    markdown = True,   
)


app = Playground(agents = [Finance_agent, web_search_agent]).get_app()

                 
if __name__=="__main__":
    serve_playground_app("Playground: app", reload = True)
    
