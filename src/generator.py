import ollama
from retriever import Retriever


class ProposalGenerator:
    def __init__(self, model="llama3"):
        self.model = model
        self.retriever = Retriever()

    def build_prompt(self, research_idea, retrieved_docs):
        context_text = "\n\n".join(
            [doc["content"] for doc in retrieved_docs]
        )

        prompt = f"""
You are an expert research grant proposal writer.

Funding Guidelines Context:
---------------------------
{context_text}

Research Idea:
--------------
{research_idea}

Generate a structured research proposal with the following sections:

1. Title
2. Abstract (300 words)
3. Problem Statement
4. Objectives (bullet points)
5. Methodology (detailed technical explanation)
6. Expected Outcomes
7. Alignment with Funding Priorities

Ensure:
- It strictly follows funding guidelines context.
- It is realistic and technically sound.
- It is suitable for submission to a government funding agency.

Return well-formatted sections.
"""
        return prompt

    def generate_proposal(self, research_idea):
        print("Retrieving funding context...")
        retrieved_docs = self.retriever.retrieve(research_idea)

        prompt = self.build_prompt(research_idea, retrieved_docs)

        print("Generating proposal...\n")

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]


if __name__ == "__main__":
    generator = ProposalGenerator()

    idea = input("Enter your research idea: ")
    proposal = generator.generate_proposal(idea)

    print("\n\n===== GENERATED PROPOSAL =====\n")
    print(proposal)