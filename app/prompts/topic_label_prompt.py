def topic_label_prompt(keywords: list[str]) -> str:
    joined = ", ".join(k for k in keywords if k.strip())

    return (
        f"Keywords: {joined}\n\n"
        "Task: Write a topic label for the keywords above.\n"
        "Rules:\n"
        "- Maximum 5 words\n"
        "- Title Case\n"
        "- No punctuation, no quotes, no explanation\n"
        "- Output the label and absolutely nothing else\n\n"
        "Label:"
    )
