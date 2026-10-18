import os

import networkx as nx
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

SCHEME_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="


def scrape_tn_schemes(url: str = SCHEME_URL) -> str:
    """Download the Tamil Nadu schemes page and extract its text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Use the content div when available; otherwise, read the whole page.
    content = soup.find("div", id="content")
    if content is None:
        content = soup

    text = content.get_text(separator="\n", strip=True)
    if not text:
        raise ValueError("The downloaded page contains no readable text.")

    return text


def process_and_build_graph(raw_text: str):
    """Split the scraped text into chunks and store them as graph nodes."""
    document = Document(
        page_content=raw_text,
        metadata={"source": SCHEME_URL},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    documents = splitter.split_documents([document])

    graph = nx.Graph()

    for index, chunk in enumerate(documents):
        node_id = f"chunk_{index}"
        graph.add_node(
            node_id,
            content=chunk.page_content,
            metadata=chunk.metadata,
        )

        if index > 0:
            graph.add_edge(
                f"chunk_{index - 1}",
                node_id,
                relation="next_chunk",
            )

    return documents, graph


class Guardrails:
    @staticmethod
    def validate_input(user_query: str) -> tuple[bool, str]:
        banned_keywords = [
            "hack",
            "exploit",
            "ignore previous instructions",
        ]
        query_lower = user_query.lower()

        if len(user_query.strip()) < 3:
            return False, "Please enter a longer question."

        if any(word in query_lower for word in banned_keywords):
            return False, "The question contains a restricted phrase."

        return True, "Passed"

    @staticmethod
    def validate_output(response_text: str) -> tuple[bool, str]:
        if not response_text or not response_text.strip():
            return False, "The model returned an empty answer."

        return True, "Passed"


def evaluate_response(
    query: str,
    answer: str,
    retrieval_context: list[str],
) -> dict:
    """Score the answer for faithfulness and relevance."""
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context,
    )

    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevance = AnswerRelevancyMetric(threshold=0.7)

    faithfulness.measure(test_case)
    relevance.measure(test_case)

    return {
        "faithfulness_score": faithfulness.score,
        "faithfulness_passed": faithfulness.is_successful(),
        "relevance_score": relevance.score,
        "relevance_passed": relevance.is_successful(),
    }


@st.cache_resource
def initialize_pipeline():
    """Scrape and prepare scheme information, then cache it."""
    raw_text = scrape_tn_schemes()
    documents, graph = process_and_build_graph(raw_text)
    return documents, graph


def clear_results():
    """Clear the question and all displayed results."""
    st.session_state.user_query = ""
    st.session_state.answer = ""
    st.session_state.eval_results = None
    st.session_state.message = ""


st.set_page_config(
    page_title="Tamil Nadu Schemes Assistant",
    layout="wide",
)

st.title("Tamil Nadu Welfare Schemes Assistant")
st.write("Ask a question about the information on the Tamil Nadu schemes page.")

# Set up saved values before creating the widgets that use them.
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "answer" not in st.session_state:
    st.session_state.answer = ""
if "eval_results" not in st.session_state:
    st.session_state.eval_results = None
if "message" not in st.session_state:
    st.session_state.message = ""

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to a .env file next to app.py.")
    st.stop()

try:
    with st.spinner("Loading scheme information..."):
        documents, graph = initialize_pipeline()
except requests.RequestException as error:
    st.error(f"Could not download the schemes page: {error}")
    st.stop()
except ValueError as error:
    st.error(str(error))
    st.stop()

st.text_input("Your question", key="user_query")

generate_col, clear_col = st.columns(2)

with generate_col:
    generate_clicked = st.button("Generate answer", use_container_width=True)

with clear_col:
    st.button(
        "Clear",
        on_click=clear_results,
        use_container_width=True,
    )

if generate_clicked:
    # Remove the previous answer before processing a new question.
    st.session_state.answer = ""
    st.session_state.eval_results = None
    st.session_state.message = ""

    user_query = st.session_state.user_query
    is_valid, message = Guardrails.validate_input(user_query)

    if not is_valid:
        st.session_state.message = message
    else:
        query_words = [
            word.lower()
            for word in user_query.split()
            if len(word) > 2
        ]

        # Find chunks that contain words from the question.
        retrieved_contexts = [
            data["content"]
            for _, data in graph.nodes(data=True)
            if any(word in data["content"].lower() for word in query_words)
        ]

        # If there are no matches, use the first three chunks.
        if not retrieved_contexts:
            retrieved_contexts = [
                document.page_content for document in documents[:3]
            ]

        retrieved_contexts = retrieved_contexts[:3]
        context_text = "\n\n".join(retrieved_contexts)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer using only the context below. "
                    "If the context does not contain the answer, say so.\n\n"
                    "Context:\n{context}",
                ),
                ("human", "{question}"),
            ]
        )

        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
        )

        try:
            with st.spinner("Generating and checking the answer..."):
                response = (prompt | model).invoke(
                    {
                        "context": context_text,
                        "question": user_query,
                    }
                )
                answer = response.content

                is_valid_output, output_message = (
                    Guardrails.validate_output(answer)
                )

                if not is_valid_output:
                    st.session_state.message = output_message
                else:
                    st.session_state.answer = answer
                    st.session_state.eval_results = evaluate_response(
                        query=user_query,
                        answer=answer,
                        retrieval_context=retrieved_contexts,
                    )

        except Exception as error:
            st.session_state.message = (
                f"Could not generate or evaluate the answer: {error}"
            )

if st.session_state.message:
    st.warning(st.session_state.message)

if st.session_state.answer:
    st.subheader("Answer")
    st.write(st.session_state.answer)

if st.session_state.eval_results:
    results = st.session_state.eval_results

    st.subheader("Answer quality checks")
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Faithfulness",
            f"{results['faithfulness_score']:.2f}",
            "Passed" if results["faithfulness_passed"] else "Failed",
        )

    with col2:
        st.metric(
            "Answer relevance",
            f"{results['relevance_score']:.2f}",
            "Passed" if results["relevance_passed"] else "Failed",
        )