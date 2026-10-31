"""Held-out evaluation of the distilled students (`heldout`)."""

# Method key (CLI name, history file <key>_history.jsonl, teacher and student directory) -> name in the paper, in the
# order the paper lists the methods.
METHODS = {
    "curriculum": "LLM-CL",
    "mcts": "MCTS",
    "llm_mcts": "LLM-MCTS",
    "muzero": "MCTS-Learn",
    "fosa_rnd": "FOSA-RND",
    "fosa_llm": "FOSA-LLM",
    "fosa": "FOSA",
}
