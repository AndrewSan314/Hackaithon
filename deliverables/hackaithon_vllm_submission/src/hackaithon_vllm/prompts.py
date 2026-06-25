def render_choices(choices: list[str]) -> str:
    return "\n".join(f"{label}. {choice}" for label, choice in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", choices))
