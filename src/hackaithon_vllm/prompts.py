def render_choices(choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(choices) > len(labels):
        raise ValueError(f"At most {len(labels)} choices are supported; got {len(choices)}.")
    return "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(choices))
