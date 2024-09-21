from typing import Literal, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers.openai_functions import PydanticOutputFunctionsParser
from langchain_core.utils.function_calling import convert_to_openai_function
import gradio as gr
from typing import Tuple
import os
import PyPDF2
from dotenv import load_dotenv
import io

import re


# Load environment variables and set OpenAI API key
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Prepare the parser and main types
class ScoreResponse(BaseModel):
    """Get the score response from the response."""

    score: float = Field(description="The numeric score between 0 and 1")
    explanation: str = Field(description="Explanation for the given score")

parser = PydanticOutputFunctionsParser(pydantic_schema=ScoreResponse)

# Define the graph state
class State(TypedDict):
    """Represents the state of the essay grading process."""
    topic: str
    context: str
    essay: str
    level: Literal["high school", "college"]
    weights: dict
    expected_word_count: int
    relevance_score: ScoreResponse
    grammar_score: ScoreResponse
    structure_score: ScoreResponse
    depth_score: ScoreResponse
    final_score: ScoreResponse
    current_node: str
    last_score: float
    
# Model reference    
llm = ChatOpenAI(model="gpt-4o-mini")

openai_functions = [convert_to_openai_function(ScoreResponse)]

# Grading functions
def check_relevance(state: State) -> State:
    """Check the relevance of the essay."""
    
    state["current_node"] = "check_relevance"
    prompt = ChatPromptTemplate.from_template(
        "Analyze the relevance of the following essay to the given topic: {topic} "
        "The essay was done with the optional context of:\n\n{context}"
        "The essay should be in the level of a {level} student."
        "Provide a relevance score between 0 and 1. "
        "Your response should start with a the score followed by the numeric score, "
        "then provide your explanation.\n\nEssay: {essay}"
    )
    try:
        chain = prompt | llm.bind(functions=openai_functions) | parser
        result = chain.invoke({ "essay": state["essay"], "topic": state["topic"], "context": state["context"], "level": state["level"] })
        print (result)
        state["relevance_score"] = result
        state["last_score"] = result.score
    except ValueError as e:
        print(f"Error in check_relevance: {e}")
        state["relevance_score"] = 0.0
    return state


def check_grammar(state: State) -> State:
    """Check the grammar of the essay."""
    
    state["current_node"] = "check_grammar"
    prompt = ChatPromptTemplate.from_template(
        "Analyze the grammar and language usage in the following essay. "
        "Take into account that the essay was done by a {level} student. "
        "Provide a grammar score between 0 and 1. "
        "Your response should start with a the score followed by the numeric score, then provide your explaination. "
        "\n\nEssay: {essay}"
    )
    try:
        chain = prompt | llm.bind(functions=openai_functions) | parser
        result = chain.invoke({ "essay": state["essay"], "level": state["level"] })
        print (result)
        state["grammar_score"] = result
        state["last_score"] = result.score
    except ValueError as e:
        print(f"Error in check_grammar: {e}")
        state["grammar_score"] = 0.0
    return state

def analyze_structure(state: State) -> State:
    """Analyze the structure of the essay."""
    
    state["current_node"] = "analyze_structure"
    prompt = ChatPromptTemplate.from_template(
        "Analyze the structure of the following essay. "
        "Take into account that the essay was done by a {level} student. "
        "Provide a structure score between 0 and 1. "
        "Your response should start with a the score followed by the numeric score, then provide your explaination."
        "\n\nEssay: {essay}"
    )
    chain = prompt | llm.bind(functions=openai_functions) | parser
    try:
        result = chain.invoke({ "essay": state["essay"], "level": state["level"] })
        state["structure_score"] = result
        state["last_score"] = result.score
    except ValueError as e:
        print(f"Error in analyze_structure: {e}")
        state["structure_score"] = 0.0
    return state

def evaluate_depth(state: State) -> State:
    """Evaluate the depth of analysis in the essay."""
    
    state["current_node"] = "evaluate_depth"
    prompt = ChatPromptTemplate.from_template(
        "Evaluate the depth of analysis in the following essay. "
        "Provide a depth score between 0 and 1. "
        "Take into account that the essay was done by a {level} student. "
        "Your response should start with a the score followed by the numeric score, then provide your explaination. "
        "\n\nEssay: {essay}"
    )
    try:
        chain = prompt | llm.bind(functions=openai_functions) | parser
        result = chain.invoke({ "essay": state["essay"], "level": state["level"] })
        state["depth_score"] = result
        state["last_score"] = result.score
    except ValueError as e:
        print(f"Error in evaluate_depth: {e}")
        state["depth_score"] = 0.0
    return state

def update_weights(state: State, weights: dict) -> State:
    """Update the weights for the scoring components."""
    
    state["weights"] = weights
    return state

def calculate_final_score(state: State) -> State:
    """Calculate the final score based on individual component scores and user-defined weights."""
    
    state["current_node"] = "calculate_final_score"
    weights = state["weights"]
    total_weight = sum(weights.values())

    final_score = (
        state["relevance_score"].score * weights["relevance"] / total_weight +
        state["grammar_score"].score * weights["grammar"] / total_weight +
        state["structure_score"].score * weights["structure"] / total_weight +
        state["depth_score"].score * weights["depth"] / total_weight
    )

    global_explanation = (
        f"Essay Evaluation Summary:\n\n"
        f"1. Relevance (Weight: {weights['relevance']}):\n"
        f"   Score: {state['relevance_score'].score:.2f}\n"
        f"   {state['relevance_score'].explanation}\n\n"
        f"2. Grammar (Weight: {weights['grammar']}):\n"
        f"   Score: {state['grammar_score'].score:.2f}\n"
        f"   {state['grammar_score'].explanation}\n\n"
        f"3. Structure (Weight: {weights['structure']}):\n"
        f"   Score: {state['structure_score'].score:.2f}\n"
        f"   {state['structure_score'].explanation}\n\n"
        f"4. Depth (Weight: {weights['depth']}):\n"
        f"   Score: {state['depth_score'].score:.2f}\n"
        f"   {state['depth_score'].explanation}\n\n"
        f"Final Score Calculation:\n"
        f"The final score is a weighted average of the above components.\n"
        f"Final Score: {final_score:.2f}\n\n"
        f"Score Breakdown:\n"
        f"- Relevance contribution: {state['relevance_score'].score * weights['relevance'] / total_weight:.2f}\n"
        f"- Grammar contribution: {state['grammar_score'].score * weights['grammar'] / total_weight:.2f}\n"
        f"- Structure contribution: {state['structure_score'].score * weights['structure'] / total_weight:.2f}\n"
        f"- Depth contribution: {state['depth_score'].score * weights['depth'] / total_weight:.2f}\n\n"
        f"Overall Assessment:\n"
        f"This essay received a final score of {final_score:.2f} out of 1.00. "
        f"The score reflects a balanced consideration of the essay's relevance to the topic, "
        f"grammatical correctness, structural coherence, and depth of analysis, "
        f"weighted according to the specified importance of each factor. "
        f"To improve, focus on the areas with lower scores as detailed above."
    )

    state["final_score"] = ScoreResponse(score=final_score, explanation=global_explanation)
    return state


# Initialize the StateGraph
workflow = StateGraph(State)


def grade_essay(topic, context, essay, level, relevance_weight, grammar_weight, structure_weight, depth_weight):
    state = State(
        topic=topic,
        context=context,
        essay=essay,
        level=level,
        weights={
            "relevance": relevance_weight,
            "grammar": grammar_weight,
            "structure": structure_weight,
            "depth": depth_weight
        },
        relevance_score={},
        grammar_score={},
        structure_score={},
        depth_score={},
        final_score={}
    )
    
    for output in app.stream(state):
        print(output)
        if output['final_score'].score is not None:
            return f"Score: {output['final_score'].score:.2f}\n\nExplanation:\n{output['final_score'].explanation}"
    
    return "Error: Failed to generate a score."

def count_words(text):
    return len(text.split())

def extract_text_from_pdf(file):
    if file is None:
        return ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Add nodes to the graph
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("check_grammar", check_grammar)
workflow.add_node("analyze_structure", analyze_structure)
workflow.add_node("evaluate_depth", evaluate_depth)
workflow.add_node("calculate_final_score", calculate_final_score)

# Define and add conditional edges
workflow.add_edge(
    "check_relevance",
    "check_grammar",
)
workflow.add_edge(
    "check_grammar",
    "analyze_structure",
)
workflow.add_edge(
    "analyze_structure",
    "evaluate_depth"
)
workflow.add_edge(
    "evaluate_depth",
    "calculate_final_score"
)

# Set the entry point
workflow.set_entry_point("check_relevance")

# Set the exit point
workflow.add_edge("calculate_final_score", END)

# Compile the graph
app = workflow.compile()


def grade_essay_for_gradio(topic: str, context: str, essay: str, level: str, 
                           relevance_weight: float, grammar_weight: float, 
                           structure_weight: float, depth_weight: float) -> Tuple[float, str]:
    state = State(
        topic=topic,
        context=context or "No context provided",  # Use a default value if context is empty
        essay=essay,
        level=level,
        weights={
            "relevance": relevance_weight,
            "grammar": grammar_weight,
            "structure": structure_weight,
            "depth": depth_weight
        },
        relevance_score={},
        grammar_score={},
        structure_score={},
        depth_score={},
        final_score={},
        current_node="",
        last_score=0.0
    )
    
    try:
        result = app.invoke(state)
        print(result)
        return result['final_score'].score, result['final_score'].explanation
    
    except Exception as e:
        # Catch any other exceptions that might occur
        print(f"An error occurred: {str(e)}")
        return 0.0, f"An unexpected error occurred: {str(e)}"

# Create the Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# Essay Grading System")
    
    with gr.Row():
        with gr.Column(scale=2):
            topic = gr.Textbox(label="Essay Topic")
            context = gr.Textbox(label="Essay Context (Optional)", lines=5, placeholder="Enter additional context for the essay (optional)")
            pdf_context = gr.File(label="Upload PDF for Context (Optional)", file_types=[".pdf"])
            essay = gr.Textbox(label="Essay Text", lines=10)
            level = gr.Radio(["high school", "college"], label="Educational Level")
            word_count = gr.Number(label="Current Word Count", value=0, interactive=False)
            expected_word_count = gr.Number(label="Expected Word Count", value=500, precision=0, step=50, interactive=True)
        
        with gr.Column(scale=1):
            gr.Markdown("## Grading Weights")
            relevance_weight = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Relevance")
            grammar_weight = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Grammar")
            structure_weight = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Structure")
            depth_weight = gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Depth")
    
    grade_button = gr.Button("Grade Essay", variant="primary")
    
    with gr.Row():
        total_score = gr.Number(label="Total Score")
        explanation = gr.Textbox(label="Detailed Explanation", lines=15)
    
    
    def update_word_count(essay_text):
        return count_words(essay_text)
    
    def update_context(pdf_file, current_context):
        if pdf_file is not None:
            pdf_text = extract_text_from_pdf(pdf_file)
            return current_context + "\n\nExtracted from PDF:\n" + pdf_text
        return current_context
    
    essay.change(update_word_count, inputs=[essay], outputs=[word_count])
    pdf_context.change(update_context, inputs=[pdf_context, context], outputs=[context])
    
    def safe_grade_essay(*args):
        try:
            score, explanation = grade_essay_for_gradio(*args)
            return score, explanation
        except Exception as e:
            return 0.0, f"An error occurred while grading the essay: {str(e)}"
        
        

    grade_button.click(
        fn=safe_grade_essay,
        inputs=[topic, context, essay, level, relevance_weight, grammar_weight, structure_weight, depth_weight],
        outputs=[total_score, explanation]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)