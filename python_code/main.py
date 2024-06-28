import easyocr
import argparse
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama


def read_text_from_image(image_path):
    """
    Read text from an image using EasyOCR.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        list: A list of tuples containing the detected text, bounding box coordinates, and confidence score.
    """
    # Create an OCR reader object
    reader = easyocr.Reader(['en'])
    
    # Read text from the image
    result = reader.readtext(image_path)
    
    return result

def return_text(text):
    """
    Print the detected text.
    
    Args:
        text (list): A list of tuples containing the detected text, bounding box coordinates, and confidence score.
    """
    txt =[]
    for detection in text:
        # if detection[2] > 0.5:
        string_list = detection[1].split()
        for word in string_list:
            print(word)
            if 'Sug' in word:
                 txt.append('Sugar')
            if 'Veg' in word:
                txt.append('Edible Vegetable oil')
                
        return txt


def main():
    """
    Extract text from an image using EasyOCR.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Extract text from an image using EasyOCR')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    # Read text from the image and print the extracted text
    text = read_text_from_image(args.image_path)
    text2 = return_text(text)
    text2 = ' & '.join(text2)
    # Load the LLM model
    llm = Ollama(model="llama2")
    
    # Load the WebBaseLoader for scrapping
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6052506/
    urls = [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6052506/",
            "https://pharmeasy.in/blog/10-harmful-effects-of-sugar/"
        ]   
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())


    embeddings_llm = OllamaEmbeddings(model="llama2") # base_url = 'http://localhost:11434'
    text_splitter = RecursiveCharacterTextSplitter()

    documents = text_splitter.split_documents(docs)
    vector_index = FAISS.from_documents(documents, embeddings_llm)
    retriever = vector_index.as_retriever()

    # relevant_docs = retriever.invoke({"input": "Check what are the side effects of Edible Vegetable oil?"})
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based on the provided context and your internal knowledge.
    Give priority to context and if you are not sure then say you are not aware of topic:

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    inp = "You are fitness influencer, make an awareness Tweet telling the side effects of " + text2 +" ?"
    response = retrieval_chain.invoke({"input":inp })

    print(response["answer"])

if __name__ == "__main__":    
    main()

