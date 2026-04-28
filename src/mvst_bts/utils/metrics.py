def format_metrics(metrics):
    return ", ".join(f"{k}={v}" for k, v in metrics.items())
