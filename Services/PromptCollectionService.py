

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

        self.compose_state_query = (
            "Compose one concise image-search query from a validated visual constraint memory. "
            "Preserve the main intent and the most useful positive constraints. Preserve negative "
            "constraints as exclusion terms using concise wording such as '-cartoon' when appropriate. "
            "Do not add new facts, do not explain, and do not mention memory/state/constraints. "
            "Use at most three positive visual details besides the main intent. Keep the query short "
            "and CLIP-friendly, usually under 16 words. Return JSON only.\n"
            "State:{state}\n"
            "JSON:{{\"rewritten_query\":\"...\"}}"
        )

        self.simulate_user_edit = (
            "Simulate a target-aware user in an interactive image retrieval session. "
            "The user knows the target image facts, sees the system suggestions and retrieved evidence, "
            "and may accept, edit, combine, or reject suggestions. Act like a real user refining a "
            "visual search query, not an oracle blindly copying all target facts. First distinguish "
            "positive constraints from negative constraints in the current query and suggestions. "
            "A negative constraint means absence/exclusion and must not be converted into a positive "
            "visual detail. Preserve useful positive and negative constraints, and remove or rewrite "
            "constraints only when they conflict with the target facts. Keep the interaction active: "
            "prefer a small target-supported edit over no-op when the state is still missing useful "
            "target details. If a suggestion is partly wrong, keep the correct part and edit the wrong "
            "part using target facts. If all suggestions are weak but the target has one clearly useful "
            "missing detail, add that detail. Use no-op only when the current state already captures "
            "the target well or when adding details would be unsupported/noisy. Produce the next search "
            "query that a careful user would submit. Use the interaction_state as memory: keep previously "
            "accepted constraints unless they are wrong, and update it with the current decision. "
            "Use only target-supported visual details. Keep the refined query concise and CLIP-friendly. "
            "Return valid JSON only.\n"
            "Context:{context}\n"
            "JSON schema:"
            "{{"
            "\"action\":\"accept|edit|combine|reject|add_detail|remove_detail\","
            "\"selected_suggestions\":[\"...\"],"
            "\"kept_constraints\":[\"...\"],"
            "\"added_constraints\":[\"...\"],"
            "\"negative_constraints\":[\"...\"],"
            "\"rejected_constraints\":[\"...\"],"
            "\"added_target_details\":[\"...\"],"
            "\"removed_details\":[\"...\"],"
            "\"refined_query\":\"...\","
            "\"reason\":\"...\""
            "}}"
        )
