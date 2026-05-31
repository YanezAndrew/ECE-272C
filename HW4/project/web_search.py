import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_search_tool = DuckDuckGoSearchResults(num_results=5, output_format="list")


def search(question: str) -> dict:
    """
    Answer a generic-domain question using DuckDuckGo web search.

    Returns
    -------
    dict with keys:
      final_answer : str   — grounded natural-language answer
      citations    : list[dict]  — each {"title": str, "url": str, "snippet": str}
    """
    # --- retrieve search results ---
    try:
        raw_results = _search_tool.invoke(question)
    except Exception as e:
        return {
            "final_answer": f"Web search failed: {e}",
            "citations": [],
        }

    # raw_results is a list of dicts with keys: snippet, title, link
    citations = []
    context_parts = []
    for r in raw_results:
        if not isinstance(r, dict):
            continue
        title = r.get("title", "")
        url = r.get("link", r.get("url", ""))
        snippet = r.get("snippet", "")
        citations.append({"title": title, "url": url, "snippet": snippet})
        context_parts.append(f"Source: {title} ({url})\n{snippet}")

    context = "\n\n".join(context_parts)

    # --- generate grounded answer ---
    system = (
        "You are a knowledgeable assistant. Answer the user's question based strictly on "
        "the web search results provided. Be concise and factual. "
        "If the search results do not contain enough information to answer confidently, "
        "say so explicitly. Do not fabricate facts."
    )
    user = f"Question: {question}\n\nWeb search results:\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    final_answer = response.choices[0].message.content.strip()

    return {
        "final_answer": final_answer,
        "citations": citations,
    }


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval-augmented generation?"
    result = search(question)
    print(f"Answer:\n{result['final_answer']}\n")
    print("Citations:")
    for c in result["citations"]:
        print(f"  - {c['title']}: {c['url']}")
