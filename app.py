import os
import json
from datetime import datetime

from loaders import load_policies
from chunking import chunking

from retrievers import (
    build_vector_store
)

from prebuilt_agent import app_graph

from config import POLICY_DATA_PATH

OUTPUT_DIR = "outputs"


def initialize_output_folder():
    """Create PRD required output files"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample_output_file = os.path.join(
        OUTPUT_DIR,
        "sample_run_outputs.md"
    )

    evaluation_file = os.path.join(
        OUTPUT_DIR,
        "evaluation_results.json"
    )

    if not os.path.exists(sample_output_file):
        with open(sample_output_file, "w", encoding="utf-8") as f:
            f.write(
                "# Enterprise Policy Assistant Sample Outputs\n\n"
            )

    if not os.path.exists(evaluation_file):
        with open(evaluation_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def save_output(question: str, answer: str):
    """Save chat output to markdown file"""

    sample_output_file = os.path.join(
        OUTPUT_DIR,
        "sample_run_outputs.md"
    )

    with open(
        sample_output_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"""
## {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Question

{question}

### Answer

{answer}

---

"""
        )


def save_evaluation(question: str, answer: str):
    """Save output to evaluation json"""

    evaluation_file = os.path.join(
        OUTPUT_DIR,
        "evaluation_results.json"
    )

    try:
        with open(
            evaluation_file,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(f)

    except Exception:
        data = []

    data.append(
        {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer
        }
    )

    with open(
        evaluation_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


def initialize():

    if not os.path.exists(POLICY_DATA_PATH):

        os.makedirs(POLICY_DATA_PATH)

        print(
            f"Put policy files into: "
            f"{POLICY_DATA_PATH}"
        )

        return False

    docs = load_policies()

    if not docs:

        print("No policy files found.")
        return False

    chunks = chunking(docs)

    build_vector_store(chunks)

    print(
        f"Indexed {len(chunks)} chunks."
    )

    return True


def chat():

    while True:

        question = input(
            "\nQuestion: "
        ).strip()

        if not question:
            continue

        if question.lower() in [
            "exit",
            "quit"
        ]:
            print("Goodbye!")
            break

        try:

            result = app_graph.invoke(
                {
                    "messages": [
                        ("user", question)
                    ]
                }
            )

            answer = result["messages"][-1].content

            print("\nAssistant:\n")
            print(answer)

            # Save outputs automatically
            save_output(question, answer)
            save_evaluation(question, answer)

        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":

    initialize_output_folder()

    if initialize():
        chat()