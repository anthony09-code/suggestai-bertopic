def topic_label_prompt(keywords: list[str]) -> str:
    prompt = f"""
            Given these keywords from a student feedback topic: {", ".join(keywords)}
            Generate a short, clear label (3-5 words) that describes this topic.
            Respond with only the label, nothing else.
            """
    return prompt
