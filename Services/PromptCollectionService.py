

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
        
        self.reason = (
            "You are a retrieval refinement assistant.\n"
            "Your task is to improve the user's image search query by proposing exactly 3 short query suggestions based on the retrieved captions.\n"
            "The goal is NOT to answer the user.\n"
            "The goal is NOT to rewrite the query with synonyms.\n"
            "The goal is to suggest 3 better follow-up queries that help the search system retrieve images closer to the user's true intention.\n\n"
            "User query: {input_query}\n"
            "Retrieved captions: {db}\n\n"
            "Rules:\n"
            "1. Use only details explicitly supported or strongly implied by the retrieved captions.\n"
            "2. Keep the user's original intent unchanged.\n"
            "3. Each suggestion must add NEW visual detail not already present in the user query.\n"
            "4. The 3 suggestions must be meaningfully different from each other.\n"
            "5. Each suggestion must focus on a different refinement aspect whenever possible, such as:\n"
            "   - action\n"
            "   - scene/background\n"
            "   - spatial relation\n"
            "   - nearby object\n"
            "   - attribute\n"
            "   - count\n"
            "6. Do NOT produce multiple suggestions with the same core meaning.\n"
            "7. If two suggestions describe the same main fact, keep only the more specific one.\n"
            "8. Prefer details that are repeated or consistent across multiple retrieved captions.\n"
            "9. If the captions are noisy or contradictory, use only the safest supported details.\n"
            "10. Do NOT hallucinate.\n"
            "11. Do NOT answer the user.\n"
            "12. Do NOT summarize the captions.\n"
            "13. Do NOT repeat or paraphrase the user query.\n"
            "14. Keep each suggestion short and directly usable as a search query.\n"
            "15. Each explanation must clearly state what NEW detail was added.\n\n"
            "Output requirements:\n"
            "- Return valid JSON only.\n"
            "- Return exactly 3 items.\n"
            "- Do not include markdown fences.\n"
            "- Do not include any text before or after the JSON.\n\n"
            "Output format:\n"
            "[\n"
            "  {{\"sug\":\"...\", \"explain\":\"...\"}},\n"
            "  {{\"sug\":\"...\", \"explain\":\"...\"}},\n"
            "  {{\"sug\":\"...\", \"explain\":\"...\"}}\n"
            "]"
        )