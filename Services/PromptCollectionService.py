

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
            "Reason carefully over the retrieved captions before answering, but do not reveal your full reasoning.\n"
            "Generate exactly 3 short follow-up search queries that help retrieve images closer to the user's true intent.\n\n"
            "User query: {input_query}\n"
            "Retrieved captions: {db}\n\n"
            "Instructions:\n"
            "1. First identify the user's main intent.\n"
            "2. Then compare the retrieved captions and find safe visual details that are explicitly supported or strongly implied.\n"
            "3. Prefer details that appear repeatedly or are consistent across multiple captions.\n"
            "4. Avoid weak, noisy, or contradictory details.\n"
            "5. Each suggestion must preserve the original intent and add one new useful visual refinement.\n"
            "6. The 3 suggestions must be meaningfully different from each other.\n"
            "7. Do not hallucinate.\n"
            "8. Keep each suggestion short and directly usable as a search query.\n"
            "9. Do not answer the user, summarize the captions, or paraphrase the original query.\n"
            "10. Do not reveal intermediate reasoning.\n\n"
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