

class PromptCollectionService:
    
    def __init__(self):
        
        self.greeting = "Hello, who are you?"
        
        self.rewrite_context = (
            "Rewrite this retrieval state as one concise image-search query.\n"
            "Keep active visual constraints, apply accepted feedback, remove rejected constraints, "
            "and include useful negatives. Do not mention dialogue/turns.\n"
            "Return JSON only: {{\"rewritten_query\":\"...\"}}\n"
            "State:{context}\n"
            "JSON:"
            "{{\"rewritten_query\":\"...\"}}"
        )
        
        self.reason = (
            "RAIR: from query and retrieved candidate evidence, give a compact diagnosis and exactly "
            "3 query-refinement suggestions. Use only supported candidate details; candidates may be "
            "distractors. Avoid weak, contradictory, or hallucinated details. Do not ask questions. "
            "Return valid JSON only. Keep all strings short; diagnosis arrays max 2 items; "
            "each explain max 12 words. Each suggestion must use keys exactly: sug,type,explain. "
            "The sug value must be a concrete image-search refinement phrase, not an instruction "
            "to the user. Do not repeat details already in Query. Do not start sug with specify, "
            "determine, clarify, ask, choose, provide, ignore, remove, exclude, or no unrelated. "
            "Avoid generic anti-distractor suggestions such as 'ignore street scene'; prefer positive "
            "target-like visual details from supported candidates.\n"
            "Query:{input_query}\n"
            "Evidence:{db}\n"
            "JSON schema:"
            "{{\n"
            "  \"diagnosis\": {{\n"
            "    \"main_intent\": \"...\",\n"
            "    \"candidate_commonalities\": [\"...\"],\n"
            "    \"candidate_ambiguities\": [\"...\"],\n"
            "    \"missing_constraints\": [\"...\"],\n"
            "    \"unsupported_details\": [\"...\"],\n"
            "    \"refinement_strategy\": \"...\"\n"
            "  }},\n"
            "  \"suggestions\": [\n"
            "    {{\"sug\":\"person lying beside a large brown dog\", \"type\":\"add_detail\", \"explain\":\"Adds supported dog detail.\"}},\n"
            "    {{\"sug\":\"person lying on a couch next to a dog\", \"type\":\"disambiguate\", \"explain\":\"Narrows the setting.\"}},\n"
            "    {{\"sug\":\"person resting indoors beside a small dog\", \"type\":\"add_detail\", \"explain\":\"Adds supported indoor detail.\"}}\n"
            "  ]\n"
            "}}"
        )

        self.compose_refinement = (
            "Compose one concise image-search query from the current query and accepted refinement. "
            "Preserve the main intent, add only useful new visual constraints, remove duplicates, "
            "and avoid meta words like suggestion/refinement. Return JSON only.\n"
            "Current query:{current_query}\n"
            "Accepted refinement:{accepted_suggestion}\n"
            "JSON:{{\"refined_query\":\"...\"}}"
        )

        self.simulate_user_edit = (
            "Simulate a target-aware user in an interactive image retrieval session. "
            "The user knows the target image facts, sees the system suggestions and retrieved evidence, "
            "and may accept, edit, combine, or reject suggestions. Produce the next search query that "
            "a careful user would submit. Use only target-supported visual details. Remove candidate "
            "details that conflict with the target. Keep the refined query concise and CLIP-friendly. "
            "Return valid JSON only.\n"
            "Context:{context}\n"
            "JSON schema:"
            "{{"
            "\"action\":\"accept|edit|combine|reject|add_detail|remove_detail\","
            "\"selected_suggestions\":[\"...\"],"
            "\"added_target_details\":[\"...\"],"
            "\"removed_details\":[\"...\"],"
            "\"refined_query\":\"...\","
            "\"reason\":\"...\""
            "}}"
        )
