import ollama


class BudgetEngine:
    def __init__(self, model="llama3"):
        self.model = model

    def build_budget_prompt(self, research_idea, duration_months, total_budget):
        prompt = f"""
You are a government research grant financial planning expert.

Research Project:
-----------------
{research_idea}

Project Duration:
-----------------
{duration_months} months

Maximum Budget:
---------------
₹{total_budget} INR

Create:

1. Month-wise timeline (milestones every 3–6 months)
2. Budget breakdown under standard Indian government categories:
   - Personnel
   - Equipment
   - Consumables
   - Travel
   - Contingency
   - Institutional Overheads
3. Ensure total does NOT exceed ₹{total_budget}
4. Provide brief justification for each category.

Return output in structured format:

TIMELINE:
Month 1–3:
Month 4–6:
...

BUDGET BREAKDOWN:
Category – Amount – Justification
"""
        return prompt

    def generate_budget(self, research_idea, duration_months, total_budget):
        prompt = self.build_budget_prompt(
            research_idea,
            duration_months,
            total_budget
        )

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]


if __name__ == "__main__":
    engine = BudgetEngine()

    idea = input("Enter research idea: ")
    duration = int(input("Enter duration (months): "))
    budget = int(input("Enter total budget (INR): "))

    result = engine.generate_budget(idea, duration, budget)

    print("\n===== BUDGET & TIMELINE =====\n")
    print(result)