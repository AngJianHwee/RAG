import os
import openai
from dotenv import load_dotenv
load_dotenv()


def create_client():
    """
    Create an OpenAI client using the API key stored in the environment
    """
    api_key = os.getenv("API_KEY")
    # default None
    base_url = os.getenv("API_BASE")

    if api_key is None:
        raise ValueError("API_KEY is not set")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    return client


def embedding_text(client, input_text):
    """
    Generate embeddings for each page using OpenAI's GPT-3

    This function assumes you have already segmented your PDF into 'pages',
    each of which is a string of text.
    """
    response = client.embeddings.create(
        input=input_text,
        model=os.getenv("EMBEDDING_MODEL")
    )
    return response.data[0].embedding


def completion_text(client, input_messages_dict):
    """
    Generate embeddings for each page using OpenAI's GPT-3

    This function assumes you have already segmented your PDF into 'pages',
    each of which is a string of text.
    """

    response = client.chat.completions.create(
        model=os.getenv("COMPLETION_MODEL"),
        messages=input_messages_dict
    )

    return str(response.choices[0].message.content)
