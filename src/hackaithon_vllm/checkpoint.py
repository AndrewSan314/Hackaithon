CKPT12_NAME = "ckpt12_internal_rag"

CHECKPOINTS = {
    CKPT12_NAME: {
        "pred_name": "pred_ckpt12_internal_rag_qlora.csv",
        "use_normalizer": True,
        "use_router": True,
        "use_router_v2": True,
        "use_router_v3": True,
        "use_bge_router": True,
        "use_bge_reading_context_route": True,
        "use_bge_context_ranker": True,
        "use_safety_solver": True,
        "use_calc_solver": True,
        "use_context_compression": True,
        "use_calc_thinking": True,
        "use_complete_option_solver": True,
        "use_answer_repair": True,
        "use_internal_rag": True,
        "use_route_lora": True,
    }
}
