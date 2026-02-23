import ollama


class ProposalEvaluator:
    def __init__(self, model="llama3"):
        self.model = model

    def build_evaluation_prompt(self, proposal_text, rubric_text=None):
        rubric_section = rubric_text if rubric_text else """
Evaluation Criteria:
1. Clarity of problem statement (20 marks)
2. Technical depth and methodology (25 marks)
3. Innovation and novelty (20 marks)
4. Feasibility and timeline realism (15 marks)
5. Budget justification (10 marks)
6. Alignment with funding priorities (10 marks)
Total: 100 marks
"""

        prompt = f"""
You are a senior government research grant reviewer.

Rubric:
-------
{rubric_section}

Proposal:
---------
{proposal_text}

Tasks:
1. Score each criterion separately.
2. Provide total score out of 100.
3. Identify major weaknesses.
4. Suggest concrete improvements.
5. State likelihood of funding (Low / Moderate / High).

Return structured output:

SCORES:
Clarity:
Technical Depth:
Innovation:
Feasibility:
Budget:
Alignment:
TOTAL:

WEAKNESSES:
- ...

IMPROVEMENTS:
- ...

FUNDING PROBABILITY:
"""
        return prompt

    def evaluate(self, proposal_text, rubric_text=None):
        prompt = self.build_evaluation_prompt(proposal_text, rubric_text)

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]


if __name__ == "__main__":
    evaluator = ProposalEvaluator()

    print("Paste proposal below (CTRL+Z then Enter to finish on Windows):")
    proposal = ""
    try:
        while True:
            proposal += input() + "\n"
    except EOFError:
        pass

    result = evaluator.evaluate(proposal)

    print("\n===== EVALUATION RESULT =====\n")
    print(result)