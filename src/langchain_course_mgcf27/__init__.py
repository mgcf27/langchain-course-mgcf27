from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    information = (
        "Ada Lovelace (1815–1852) was an English mathematician and writer. "
        "She worked with Charles Babbage on his proposed Analytical Engine. "
        "Her notes included an algorithm for calculating Bernoulli numbers, "
        "and she recognized that the machine could manipulate symbols beyond numbers."
    )
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=(
            "Given the following information about a person:\n{information}\n\n"
            "Write a short summary and two interesting facts about that person."
        ),
    )
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
        base_url="http://127.0.0.1:11434",
    )
    chain = summary_prompt_template | llm
    print(type(llm))
    response = chain.invoke({"information": information})
    print("CONTENT:", response.content)

