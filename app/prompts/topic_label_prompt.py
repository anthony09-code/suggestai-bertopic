def topic_label_prompt(keywords: list[str]) -> str:
    joined = ", ".join(k for k in keywords if k.strip())
    return (
        f"You are labeling a cluster of student feedback topics from a university.\n"
        f"Keywords from this cluster: {joined}\n\n"
        "Write a short, meaningful label that captures the theme of these keywords.\n"
        "Rules:\n"
        "- Maximum 4 words\n"
        "- Title Case\n"
        "- Must be a noun phrase describing the theme (e.g. 'Teaching Effectiveness', 'Workload Concerns')\n"
        "- No punctuation, no quotes, no explanation, no reasoning\n"
        "- If keywords are in Filipino/Tagalog, still write the label in English\n"
        "- Output only the label, nothing else\n\n"
        "Label:"
    )
