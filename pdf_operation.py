from langchain_community.document_loaders import PyPDFLoader
from client import embedding_text


def load_pdf(object_storage_location):
    """
    Load a PDF file using the PyPDFLoader
    """
    loader = PyPDFLoader(object_storage_location)
    pages = loader.load_and_split()
    return pages


def generate_embeddings(pages, openai_client):
    """
    Generate embeddings for each page using OpenAI's GPT-3

    This function assumes you have already segmented your PDF into 'pages',
    each of which is a string of text.
    """
    embeddings = []

    for page in pages:
        # get content
        page_content = page.page_content
        embeddings.append(
            {
                "text": page_content,
                "embedding": embedding_text(openai_client, page_content)
            }
        )

    return embeddings
