from generator import ProposalGenerator
from evaluator import ProposalEvaluator


def critique_loop(research_idea):
    generator = ProposalGenerator()
    evaluator = ProposalEvaluator()

    print("Generating initial proposal...\n")
    proposal = generator.generate_proposal(research_idea)

    print("\nEvaluating proposal...\n")
    evaluation = evaluator.evaluate(proposal)

    print("\n===== EVALUATION =====\n")
    print(evaluation)

    print("\nImproving proposal based on feedback...\n")

    improvement_prompt = f"""
Improve the following proposal based on reviewer feedback.

Proposal:
---------
{proposal}

Reviewer Feedback:
------------------
{evaluation}

Generate improved full proposal.
"""

    improved_proposal = generator.model
    import ollama
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": improvement_prompt}]
    )

    print("\n===== IMPROVED PROPOSAL =====\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    idea = input("Enter research idea: ")
    critique_loop(idea)