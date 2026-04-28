def cosine_warmup_scheduler(total_steps, warmup_steps):
    return {
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
    }
