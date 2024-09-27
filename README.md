# Retrieval-Augmented Generation with Summarization Agent using text based documents

## Table of Contents
- [Overview](#overview)
- [API Key](#api-key)
- [Architecture](#architecture)
- [Data Ingestion Notebook](#data-ingestion-notebook)
  - [1. Document Loading](#1-document-loading)
  - [2. Document Splitting](#2-document-splitting)
  - [3. Embeddings and Vectorstores](#3-embeddings-and-vectorstores)
  - [4. Retrieval](#4-retrieval)
  - [5. Simple Chatbot](#5-simple-chatbot)
- [Chatbot Script](#chatbot-script)
  - [Setup](#setup)
  - [Chat History](#chat-history)
  - [Response Handling](#response-handling)
  - [Summarization](#summarization)
- [Use of Streamlit for the chatbot UI](#use-of-streamlit-for-the-chatbot-ui)
- [Requirements](#requirements)
- [Costs Involved](#costs-involved)
- [Technical Sustainability](#technical-sustainability)
- [Contact Information](#contact-information)

## Overview
This project involves the development of a chatbot that utilizes LangChain and Retrieval-Augmented Generation (RAG) models to provide intelligent responses based on a collection of documents. The following sections will guide you through the main components of the project, the data ingestion, and the chatbot script.

### What is LangChain?
LangChain is a framework designed to simplify the integration of large language models (LLMs) into applications. It provides tools for handling various tasks such as text embedding, document retrieval, and chatbot interactions. By leveraging LangChain, developers can efficiently manage LLMs and enhance their applications with advanced language capabilities.

### What are RAG Models?
Retrieval-Augmented Generation (RAG) models combine the strengths of retrieval-based and generation-based approaches. In a RAG model, a retriever first selects relevant documents or text chunks from a corpus, and then a generator (typically an LLM) produces a coherent response based on the retrieved content. This method improves the relevance and accuracy of generated responses, making it ideal for applications like chatbots and question-answering systems.

## API Key

The use of an API key is crucial for accessing OpenAI's services. The API key is used to:

- **Authenticate Requests**: Ensures that the application can access OpenAI's models and services.
- **Create the Embed Text**: Utilizes OpenAI's embedding models to convert text into vector representations.
- **Generate Responses**: Allows the LLM to generate responses based on retrieved documents.


## Files
![Files and Directories](diagram/Screenshot%202024-07-22%20200518.png)

## Architecture

![Chatbot diagram](diagram/UC2%20chatbot.png)

## Data Ingestion Notebook

### 1. Document Loading
   
The document loading process involves reading text from PDF and DOCX files and extracting their content along with metadata. Here's a step-by-step explanation:

- **Load DOCX Files**: A custom loader class (DocxLoader) is used to extract text and metadata from DOCX files.
- **Load PDF Files**: The extract_text_from_pdf function utilizes the PyPDF2 library to extract text from PDF files.
- **Metadata Extraction**: For each file, metadata such as the file title and page numbers are extracted and associated with the text content.
- **Directory Traversal**: The script iterates through a specified directory, loading all PDF and DOCX files, extracting their content, and storing them in a list of documents.

### 2. Document Splitting

To manage large documents, the text is split into smaller chunks using the `RecursiveCharacterTextSplitter` from LangChain. This step involves:

- **Initialization**: Setting parameters for chunk size, overlap, and separators.
- **Splitting**: The text from each document is split into manageable chunks, and metadata is updated to include the split number.
- **Validation**: The script prints some of the split chunks to verify the process and calculates the average chunk length.

Note: chunk length is an important metric, as larger chunks will increase the tokens used by the LLM (and thus increase the cost) while smaller chunks contain less context.

### 3. Embeddings and Vectorstores

This step involves embedding the text chunks and storing them in a vector store for efficient retrieval:

- **Embeddings**: Using OpenAI's `text-embedding-ada-002` model, the text chunks are converted into embeddings.
- **Vector Store Creation**: The Chroma vector store is used to store the embeddings along with their metadata.
- **Batch Processing**: Documents are processed in batches, and the vector store is saved locally for later use.

### 4. Retrieval

The retrieval process involves setting up a retriever to fetch relevant documents based on a query:

- **Self-Query Retriever**: This retriever uses an LLM to interpret and process queries, retrieving relevant documents from the vector store.
- **Retrieval methods**: Similarity search identifies and retrieves the most relevant documents or text chunks by comparing their embeddings with the query embedding, ranking them based on similarity scores. Maximal Marginal Relevance (MMR) enhances this process by balancing relevance and diversity, ensuring the retrieved chunks are not only relevant but also diverse. MMR minimizes redundancy by selecting chunks that provide new information compared to previously selected ones. This combination results in a more diverse set of search results.

**Chunk Selection**:
The chunk selection process involves the following steps:

1. **Embedding Calculation**: Both the query and the document chunks are converted into vector embeddings using the same model.
2. **Similarity Scoring**: The cosine similarity between the query embedding and each document chunk embedding is calculated.
3. **Ranking**: Chunks are ranked based on their similarity scores.
4. **MMR Application**: If MMR is used, chunks are selected to maximize relevance while ensuring diversity in the retrieved set.
5. **Result Compilation**: The top chunks are compiled and returned as the retrieval result.

### 5. Simple Chatbot

A very basic conversational chatbot is created using LangChain's `ConversationalRetrievalChain`:

1. **Prompt Template**: Defines how the chatbot should respond, including context and question formatting.
2. **Memory**: Uses ConversationBufferMemory to maintain the context of the conversation.
3. **Chain Setup**: Combines the LLM, retriever, and prompt template to create the conversational chain.
4. **Example Interaction**: Demonstrates how the chatbot can answer questions based on the retrieved documents.

## Chatbot Script

### Setup
- **Initialization**: The script initializes OpenAI embeddings and the vector store saved during ingestion.
- **Retriever Configuration**: A retriever is configured to fetch relevant documents from the vector store using a MMR search.
### Chat History
- **Maintaining Context**: The script uses `ConversationBufferMemory` to keep track of the chat history.
### Response Handling
- **User Input**: The script handles user input, retrieves relevant information from the vector store, and generates concise responses using the LLM.
- **Generated Responses**: The responses are designed to be brief and relevant, with a structure that provides direct answers based on the retrieved documents.
### Summarization
The summarization agent in the script serves to condense the information from documents into concise summaries upon user request as the base RAG model is not suited for full document summarization:

- **Document Retrieval**: The agent first retrieves the most relevant document using similarity search.
- **Filtered Retrieval**: It then performs another retrieval to get a broader context around the document title (gets all chunk that belong to that document).
- **Prompt Definition**: A specific prompt template is used to instruct the LLM to generate a summary.
- **LLM Chain**: An LLMChain is created using the prompt, and a `StuffDocumentsChain` is used to compile and process the documents into a coherent summary.
- **Functionality**: The summarization agent detects requests for summaries based on keywords in the user's query. If such a request is identified, it summarizes the document and provides the user with a concise summary along with the source document's title.

## Use of Streamlit for the chatbot UI

- **User Interface**: Provides an interactive web interface for the chatbot. Users can input their queries and view responses in a chat-like format.
- **Input Handling**: Captures user input through an input box and processes it accordingly.
- **Chat Display**: Displays the chat history, ensuring users can see previous interactions, enhancing the conversational context.
- **Sessions**: Sessions can be configured to return a past discussion.

## Requirements

The requirements.txt file includes all necessary packages.

## Costs Involved

For this project access to openai's API is needed. For that I added credits in a pay-as-you-go subcription. All embeddings, testing, and LLM usage over the project duration costs a total of ~$7 CAD.
We can expect the costs to be much higher when needing to ingest a larger number of documents as well as increased LLM use but ultimately the costs are quite low and token dependent.

The official pricing for the services used are (as of july 2024):
- gpt-3.5-turbo: $0.50 / 1M input tokens - $1.50 / 1M output tokens
- ada v2: $0.10 / 1M tokens

A regular API key will work for both models.

## Technical Sustainability

For this project to be put into production, the embedding would have to be hosted on a vector database.

- 1st Solution (simplest): Using Docker to create a Chroma DB service
- 2nd Solution (robust): Creating a vector storage database (requires more setup)

### Using Chroma (Recommended):
This approach is easier as we are already using Chroma in our local solution. We would need to dockerize it:

- Chroma is the open source vector store used for the chatbot: https://docs.trychroma.com/
- Here is an article on how to create a docker image for the Chroma DB vector store: https://abhishektatachar.medium.com/run-chroma-db-on-a-local-machine-and-as-a-docker-container-a9d4b91d2a97
- Chroma documentation on AWS deployment (easiest option): https://docs.trychroma.com/deployment/aws

This method requires minimal code edits as using another db would require a lot of code adaptation.

### Using another database:
This approach is if WVC needs to create a full vector store database. All ingestion and query sections will have to fit the new database.

- Follow this documentation to handle ingestion and querying using Azure Cosmo DB as a vector store: https://python.langchain.com/v0.1/docs/integrations/vectorstores/azure_cosmos_db/
- other vector stores: https://python.langchain.com/v0.1/docs/integrations/vectorstores/

Edits would have to be made to the following sections:
- In the ingestion notebook: When creating and loading the vector store
- In the chatbot script when using the retriever and loading 

## Contact Information

For any questions or concerns, please contact:
- Name: [Valentin Najean]
- Email: [najeanvalentin@gmail.com]
- GitHub: https://github.com/valnaj
