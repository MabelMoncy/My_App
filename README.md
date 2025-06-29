#  Algorithm Thinking with Python AI assistant
#⚠️ Project is under development!!!
This app is specially build for KTU 1st year students of 2024 scheme. This web app help them to learn the new subject Algorithm Thinking with Python with the help of AI.
(The pdf file can be changed and another pdf can be added. eg., You can add chemistry.pdf and ask questions based on that pdf.)
Only edit the Python code and change the UI.

This app is now ready to use and has been successfully deployed online. The app.py file serves as the official source code and foundational structure of the application. All necessary deployment procedures have been completed, and the app is now live and accessible for use.
- Use the app: https://alpy.onrender.com

It has been developed primarily as a learning exercise to understand the working of APIs—particularly OpenAI APIs—and their integration, API calls, Testing platforms like postman, etc. However, the project may still be useful for students interested in learning about coding and API implementation.

This is a simple yet powerful AI web application that answers user queries based on preloaded content. Built using OpenAI's API, it provides intelligent and context-aware responses to help users interact with documents more efficiently.

##  Features

-  AI-powered Q&A system  
-  Works on preloaded PDF documents  (You can add you own pdf )
-  Fast and interactive UI  
-  Uses OpenAI models with ChromaDB for vector search  

##  Tech Stack

- Python (Backend)
- Streamlit (Frontend)
- OpenAI API (LLMs)
- ChromaDB (Vector Store)
- LangChain (for orchestration)

## How to use the app locally in your device?.

Before you begin, you need to create an account on OpenAI and generate an API key. Save this key in a .env file located in the root directory of your project in the following format:
OPENAI_API_KEY=your_openai_api_key_here.
The OpenAI API offers a wide range of powerful capabilities. It’s highly recommended to explore the official documentation to understand its full potential and best practices.

- Make a folder named 'Ai_app' or whatever you like.
- Open it using an IDE (VS Code).
- Open terminal >> create a virtual environment ( python -m  venv myvenv ) >> activate it ( myvenv/Scripts/activate )
- Install the libraries.(pip install openai streamlit python-dotenv PyPDF2 chromadb langchain)
- Create a file app.py
- copy the code from app.py
- Execute the file >> streamlit run app.py
- If any error is found solve it
- Your Ai assistant is ready to use.

 # Deployment Status
 The application is hosted on Render.com, a modern cloud platform that supports full-stack web apps with powerful deployment capabilities. We chose Render for its: Seamless integration with GitHub, Support for Python and Streamlit, Easy management of environment variables (e.g., API keys), Compatibility with required dependencies like chromadb, pypdf, and langchain

# Environment Management
Sensitive information like the OpenAI API key is securely stored using Render's Environment Variables, ensuring that credentials are never exposed in the source code.
