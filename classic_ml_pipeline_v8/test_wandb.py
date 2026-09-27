from config import config
from tracking import finish_tracking, log_metrics, log_summary, start_tracking


if __name__ == "__main__":
    # Force W&B on for this connectivity test without changing config.py permanently.
    config.tracking.wandb = True
    config.tracking.wandb_mode = "online"
    original_name = str(config.general.experiment_name)
    config.general.experiment_name = f"{original_name}_wandb_test"

    run = start_tracking(config)
    if run is None:
        raise RuntimeError("W&B test run did not start. Check `wandb login` and tracking config.")

    log_metrics(config, {"test/value": 1.0}, step=0)
    log_summary(config, {"test_status": "ok"})
    finish_tracking(config)
    print("W&B test run sent successfully.")
