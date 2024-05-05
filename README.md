# Medical Chatbot - Llama 2 & Chainlit

## Overview
This project implements a medical chatbot utilizing the Llama 2 language model and Chainlit for conversational AI capabilities. The chatbot is designed to assist users with medical queries, offering information, guidance, and support across various health concerns.

## Features
- **Natural Language Understanding**: Leveraging Llama 2, the chatbot comprehends natural language queries, enabling users to interact conversationally.
- **Medical Knowledge Base**: Supported by a robust medical knowledge base, the chatbot delivers accurate and reliable information on a wide array of health topics.


## Getting Started
To utilize the medical chatbot, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies as specified in the `requirements.txt` file.
3. Configure the Faiss vector store path in the code (`DB_FAISS_PATH` constant).
4. Run the main script to initialize the chatbot.
5. Interact with the chatbot by providing natural language queries or describing symptoms.

## Flow Description
The codebase follows a structured flow:
1. **Imports and Constants**: Necessary modules are imported, and constants are set.
2. **Custom Prompt Template**: A template for question-answer retrieval is defined.
3. **Custom Prompt Configuration**: Configuration of a custom prompt template using `PromptTemplate`.
4. **Loading Language Model (LLM)**: Initialization of the Llama 2 language model with specified parameters.
5. **Retrieval QA Chain Configuration**: Configuration of a retrieval-based question answering chain using LLM and Faiss vector store.
6. **Initializing the QA Bot**: Initialization of the question-answering bot with embeddings, Faiss vector store, LLM, and a custom prompt.
7. **Handling User Queries**: Functions to handle user queries and retrieve responses from the bot.
8. **Chainlit Integration**: Integration with Chainlit, including functions to start the bot and handle user messages.

## Technologies Used
- **Llama 2**: State-of-the-art language model for natural language understanding and generation.
- **Chainlit**: Conversational AI platform for building interactive chatbots with advanced capabilities.
- **Python**: Primary programming language used for development.

## Contributing
Contributions to the project are welcome! If you have ideas for improvements or new features, feel free to fork the repository, make your changes, and submit a pull request. Please adhere to the existing coding style and guidelines.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- The Llama 2 development team for providing an excellent language model.
- The Chainlit team for their conversational AI platform.
- Contributors to open-source libraries and tools used in this project.


