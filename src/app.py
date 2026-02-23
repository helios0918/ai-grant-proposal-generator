import streamlit as st
import os
from parser import extract_text_from_pdf
from text_chunker import chunk_text
from embedder import Embedder
from retriever import Retriever
from generator import ProposalGenerator
from budget_engine import BudgetEngine
from evaluator import ProposalEvaluator
import numpy as np
import faiss
import pickle

st.set_page_config(page_title="AI Grant Proposal System", layout="wide")

st.title("AI-Based Research Grant Proposal Generator & Evaluator")

st.sidebar.header("Project Inputs")

research_idea = st.sidebar.text_area("Enter Research Idea")
duration = st.sidebar.number_input(
    "Project Duration (Months)",
    min_value=6,
    max_value=60,
    value=24
)
budget = st.sidebar.number_input(
    "Total Budget (INR)",
    min_value=100000,
    max_value=50000000,
    value=2000000
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Funding Call PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    os.makedirs("data/temp", exist_ok=True)
    file_path = os.path.join("data/temp", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    embedder = Embedder()
    embeddings = embedder.embed_texts(chunks)
    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/faiss.index")

    metadata = [{"source": uploaded_file.name, "content": c} for c in chunks]
    with open("vector_store/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    st.success("Funding Call Processed Successfully!")

if st.button("Generate Proposal"):
    if research_idea.strip():
        generator = ProposalGenerator()
        proposal = generator.generate_proposal(research_idea)
        st.session_state["generated_proposal"] = proposal
        st.subheader("Generated Proposal")
        st.write(proposal)
    else:
        st.warning("Please enter research idea.")

if st.button("Generate Budget & Timeline"):
    if research_idea.strip():
        budget_engine = BudgetEngine()
        result = budget_engine.generate_budget(
            research_idea,
            duration,
            budget
        )
        st.subheader("Budget & Timeline")
        st.write(result)
    else:
        st.warning("Enter research idea first.")

if st.button("Evaluate Proposal"):
    if "generated_proposal" in st.session_state:
        evaluator = ProposalEvaluator()
        evaluation = evaluator.evaluate(
            st.session_state["generated_proposal"]
        )
        st.session_state["latest_evaluation"] = evaluation
        st.subheader("Evaluation Result")
        st.write(evaluation)
    else:
        st.warning("Generate proposal first.")

if st.button("Improve Proposal"):
    if "generated_proposal" in st.session_state:
        import ollama

        proposal_text = st.session_state["generated_proposal"]

        reviewer_feedback = st.session_state.get(
            "latest_evaluation",
            "No formal evaluation provided. Improve clarity, innovation, feasibility, methodology strength, and impact alignment with funding priorities."
        )

        improvement_prompt = f"""
Improve the following proposal based on reviewer feedback.

Proposal:
---------
{proposal_text}

Reviewer Feedback:
------------------
{reviewer_feedback}
"""

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "user", "content": improvement_prompt}
            ]
        )

        improved_proposal = response["message"]["content"]

        st.subheader("Improved Proposal")
        st.write(improved_proposal)

        st.session_state["generated_proposal"] = improved_proposal

    else:
        st.warning("Generate proposal first.")