

class PromptCollectionService:
    
    def __init__(self):
        
        self.greeting = "Hello, who are you?"
        
        self.convert_triplet = (
            "You are a Vision-Language Model.\n"
            "User question: Convert the given sentence into (subject, relation, object) triplet.\n"
            "Rules:\n"
            "- Do NOT add explanations.\n"
            "- Use lowercase for relation.\n"
            "Return format:\n"
            "[{{\"subject\":\"...\",\"relation\":\"...\",\"object\":\"...\"}}]\n"
            "Sentence:\n"
            "{text}"
        )

        self.rewrite_context = (
            "You are a query rewriting assistant for interactive image retrieval.\n"
            "Rewrite the current retrieval context into one natural-language image search query.\n\n"
            "Context JSON:\n"
            "{context}\n\n"
            "Instructions:\n"
            "1. Preserve the user's initial intent.\n"
            "2. Apply the user's feedback to previous suggestions.\n"
            "3. Keep active visual constraints and remove rejected ones.\n"
            "4. Include negative constraints only when visually useful, such as 'no people'.\n"
            "5. Do not mention the dialogue, suggestions, turns, or feedback process.\n"
            "6. Return valid JSON only.\n\n"
            "Output format:\n"
            "{{\"rewritten_query\":\"...\"}}"
        )
        
        self.reason = (
            "You are a retrieval refinement assistant.\n"
            "Reason carefully over the retrieved captions before answering, but do not reveal your full reasoning.\n"
            "Generate exactly 3 short follow-up search queries that help retrieve images closer to the user's true intent.\n\n"
            "User query: {input_query}\n"
            "Structured triplets: {triplets}\n"
            "Retrieved captions: {db}\n\n"
            "Instructions:\n"
            "1. First identify the user's main intent.\n"
            "2. Use the structured triplets as constraints for reasoning, not as text to copy verbatim.\n"
            "3. Then compare the retrieved captions and find safe visual details that are explicitly supported or strongly implied.\n"
            "4. Prefer details that appear repeatedly or are consistent across multiple captions.\n"
            "5. Avoid weak, noisy, contradictory, or constraint-breaking details.\n"
            "6. Each suggestion must preserve the original intent and add one new useful visual refinement.\n"
            "7. The 3 suggestions must be meaningfully different from each other.\n"
            "8. Do not hallucinate.\n"
            "9. Keep each suggestion short and directly usable as a search query.\n"
            "10. Do not answer the user, summarize the captions, or paraphrase the original query.\n"
            "11. Do not reveal intermediate reasoning.\n\n"
            "Output requirements:\n"
            "- Return valid JSON only.\n"
            "- Return exactly 3 items.\n"
            "- No markdown fences.\n"
            "- No extra text.\n\n"
            "Output format:\n"
            "[\n"
            "  {{\"sug\":\"...\", \"explain\":\"...\"}},\n"
            "  {{\"sug\":\"...\", \"explain\":\"...\"}},\n"
            "  {{\"sug\":\"...\", \"explain\":\"...\"}}\n"
            "]"
        )
