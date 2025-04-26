### Imports

# Gaia Agent
from gaia_agent import GAIAAgent

# Misc
import os
import json


### Testing

with open(
    os.path.join(os.path.dirname(__file__), "resources", "benchmark_questions.json"),
    encoding="utf8",
) as f:
    test_cases = json.load(f)

    for test_case in test_cases:
        question = test_case["question"]
        level = test_case["Level"]
        file_name = test_case["file_name"]

        print(f"Question: {question}\nLevel: {level}\nFile: {file_name}")

        # Initialize the agent
        gaia_agent = GAIAAgent()

        # Run the agent with the test case question
        response = gaia_agent(question)

        # Check final answer
        print(f"Final Answer: {response}\n")
