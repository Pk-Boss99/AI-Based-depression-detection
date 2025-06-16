# File: modules/text_analysis/question_sets.py

# Scientifically validated question sets for text input.

# Patient Health Questionnaire-9 (PHQ-9) questions
# Note: The standard PHQ-9 asks about the last 2 weeks and has specific scoring
# based on frequency (not text content). Here, we use the questions as prompts
# for free-text responses to be analyzed by NLP.
PHQ9_QUESTIONS = [
    {
        "id": "phq9_1",
        "text": "Little interest or pleasure in doing things?"
    },
    {
        "id": "phq9_2",
        "text": "Feeling down, depressed, or hopeless?"
    },
    {
        "id": "phq9_3",
        "text": "Trouble falling or staying asleep, or sleeping too much?"
    },
    {
        "id": "phq9_4",
        "text": "Feeling tired or having little energy?"
    },
    {
        "id": "phq9_5",
        "text": "Poor appetite or overeating?"
    },
    {
        "id": "phq9_6",
        "text": "Feeling bad about yourself—or that you are a failure or have let yourself or your family down?"
    },
    {
        "id": "phq9_7",
        "text": "Trouble concentrating on things, such as reading a newspaper or watching television?"
    },
    {
        "id": "phq9_8",
        "text": "Moving or speaking so slowly that other people could have noticed? Or the opposite—being so fidgety or restless that you have been moving around a lot more than usual?"
    },
    {
        "id": "phq9_9",
        "text": "Thoughts that you would be better off dead, or of hurting yourself in some way?"
    }
]

# You can add other question sets here in the future
# EXAMPLE_SET_QUESTIONS = [
#     {"id": "ex_1", "text": "How do you feel today?"},
#     {"id": "ex_2", "text": "Describe your mood."}
# ]

# A dictionary to easily access question sets by name
QUESTION_SETS = {
    "PHQ-9": PHQ9_QUESTIONS,
    # "EXAMPLE_SET": EXAMPLE_SET_QUESTIONS,
}

# Default question set to use
DEFAULT_QUESTION_SET = "PHQ-9"

# You might want a function to get the questions
def get_questions(set_name: str = DEFAULT_QUESTION_SET) -> list:
    """
    Retrieves a list of questions for a given set name.

    Args:
        set_name: The name of the question set (e.g., "PHQ-9").

    Returns:
        A list of dictionaries, where each dictionary represents a question
        with 'id' and 'text' keys. Returns an empty list if the set name is invalid.
    """
    return QUESTION_SETS.get(set_name, [])

# Example Usage (for testing)
if __name__ == "__main__":
    phq9 = get_questions("PHQ-9")
    print(f"Loaded {len(phq9)} PHQ-9 questions:")
    for q in phq9:
        print(f"- {q['id']}: {q['text']}")

    invalid_set = get_questions("InvalidSet")
    print(f"\nLoaded {len(invalid_set)} questions for InvalidSet.")
