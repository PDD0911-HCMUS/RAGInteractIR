

class PromptCollectionService:
    
    def __init__(self):
        
        self.greeting = "Hello, who are you?"
        
        self.rewrite_context = (
            "Rewrite this retrieval state as one concise image-search query.\n"
            "Keep active visual constraints, apply accepted feedback, remove rejected constraints, "
            "and preserve explicit negative constraints only when they are present in the state. "
            "Do not invent generic exclusions such as cartoon, drawing, illustration, photo, or digital. "
            "Do not mention dialogue/turns.\n"
            "Return JSON only: {{\"rewritten_query\":\"...\"}}\n"
            "State:{context}\n"
            "JSON:"
            "{{\"rewritten_query\":\"...\"}}"
        )
        
        self.reason = (
            "RAIR: inspect the query and candidate evidence, then suggest concise visual refinements "
            "that could help an image search user narrow the result. Use only details supported by "
            "candidate evidence. Candidates may be distractors, so prefer details that are visually "
            "specific but not overly risky. Do not ask questions. Return valid JSON only. "
            "Give exactly 3 suggestions; each suggestion must be a search phrase, not an instruction. "
            "Avoid repeating details already present in Query unless adding a new visual detail.\n"
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
            "Rewrite the validated visual memory into one short image-search query. Keep the main "
            "intent and only the most useful visual details. Preserve existing negative constraints, "
            "but do not invent new ones. Do not add facts. Return JSON only.\n"
            "State:{state}\n"
            "JSON:{{\"rewritten_query\":\"...\"}}"
        )

        self.simulate_user_edit = (
            "Simulate a user refining an image search. The user knows the target facts and sees RAIR "
            "suggestions. Choose a small useful edit when a suggestion or missing target fact can improve "
            "the query. Edit partially correct suggestions instead of blindly accepting them. Reject/no-op "
            "when suggestions are unsupported or the current state is already good. Keep useful previous "
            "constraints from interaction_state. Use only target-supported details. Return valid JSON only.\n"
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
