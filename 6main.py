import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


load_dotenv()

MAX_ATTEMPTS = 3


class State(TypedDict):
    ingredients: str
    recipe: str
    feedback: str
    approved: bool
    attempts: int


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0
)


def get_text(response):
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text") and part.text:
                parts.append(part.text)

        return "\n".join(parts)

    return str(content)


def generate_recipe(state: State):

    prompt = f"""
    Create a simple and delicious recipe
    using these ingredients:

    {state["ingredients"]}

    Include:
    - Recipe name
    - Ingredient quantities
    - Preparation steps
    - Cooking time

    Do not invent ingredients that are not provided.
    """

    response = llm.invoke(prompt)
    recipe = get_text(response)

    print("\n--- GENERATED RECIPE ---")
    print(recipe)

    return {
        "recipe": recipe,
        "attempts": 1
    }


def check_recipe(state: State):

    prompt = f"""
    Check the quality of this recipe.

    Ingredients:
    {state["ingredients"]}

    Recipe:
    {state["recipe"]}

    Check:
    - Does it use the provided ingredients?
    - Are the instructions clear?
    - Are the ingredient quantities reasonable?
    - Is the recipe complete?

    If the recipe is good, write:
    APPROVED: TRUE

    If the recipe needs improvement, write:
    APPROVED: FALSE

    Then write:
    FEEDBACK: your explanation
    """

    response = llm.invoke(prompt)
    result = get_text(response)

    print("\n--- RECIPE CHECK ---")
    print(result)

    match = re.search(
        r"APPROVED\s*:\s*(TRUE|FALSE)",
        result,
        re.IGNORECASE
    )

    approved = (
        match is not None
        and match.group(1).upper() == "TRUE"
    )

    feedback_match = re.search(
        r"FEEDBACK\s*:\s*(.*)",
        result,
        re.IGNORECASE | re.DOTALL
    )

    feedback = (
        feedback_match.group(1).strip()
        if feedback_match
        else result
    )

    return {
        "approved": approved,
        "feedback": feedback
    }


def improve(state: State):

    prompt = f"""
    Improve the following recipe using the feedback.

    Ingredients:
    {state["ingredients"]}

    Current recipe:
    {state["recipe"]}

    Feedback:
    {state["feedback"]}

    Create a better version of the recipe.
    Use only the provided ingredients.
    Include ingredient quantities, preparation steps,
    and cooking time.
    """

    response = llm.invoke(prompt)
    improved_recipe = get_text(response)

    print("\n--- IMPROVED RECIPE ---")
    print(improved_recipe)

    return {
        "recipe": improved_recipe,
        "attempts": state["attempts"] + 1
    }


def route_after_check(state: State):

    if state["approved"]:
        return "end"

    if state["attempts"] >= MAX_ATTEMPTS:
        return "end"

    return "improve"


graph_builder = StateGraph(State)

graph_builder.add_node("generate_recipe", generate_recipe)
graph_builder.add_node("check_recipe", check_recipe)
graph_builder.add_node("improve", improve)

graph_builder.add_edge(START, "generate_recipe")
graph_builder.add_edge("generate_recipe", "check_recipe")
graph_builder.add_edge("improve", "check_recipe")

graph_builder.add_conditional_edges(
    "check_recipe",
    route_after_check,
    {
        "improve": "improve",
        "end": END
    }
)

graph = graph_builder.compile()


initial_state = {
    "ingredients": "chicken, rice, onion, tomato",
    "recipe": "",
    "feedback": "",
    "approved": False,
    "attempts": 0
}

result = graph.invoke(initial_state)

print("\n==============================")
print("FINAL RECIPE")
print("==============================")

print(result["recipe"])

print("\nApproved:", result["approved"])
print("Attempts:", result["attempts"])
print("Feedback:", result["feedback"])