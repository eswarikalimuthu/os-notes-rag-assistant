"""
cli.py
------
Quick terminal demo of the RAG pipeline, useful for testing without
running the API server.

Run:
    python cli.py
"""

#import rag          # paid: uses Anthropic Claude API
import rag_free as rag   # free: uses local Ollama instead -- comment the line above
                            # and uncomment this one to run with zero API cost

def main():
    print("OS Notes RAG Assistant (type 'exit' to quit)\n")
    while True:
        query = input("Ask> ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        result = rag.ask(query)

        print("\n--- Retrieved chunks ---")
        for s in result["sources"]:
            print(f"  {s['id']:<28} ({s['title']})  distance={s['distance']:.4f}")

        print("\n--- Answer ---")
        print(result["answer"])
        print()


if __name__ == "__main__":
    main()
