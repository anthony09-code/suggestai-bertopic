def topic_label_prompt(keywords: list[str]) -> str:
    prompt = f"Based on the following keywords, generate a concise topic label:\n\n{', '.join(keywords)}"
    return prompt
