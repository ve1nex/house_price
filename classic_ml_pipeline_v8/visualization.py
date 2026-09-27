from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_validation_plot(y_true, y_pred, config) -> None:
    if not bool(config.visualization.save_validation_plot) or len(y_true) == 0:
        return
    output_dir = Path(config.paths.path_to_plots)
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(config.general.task) == "classification":
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
        ax.set_title("OOF confusion matrix")
    else:
        y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
        residual = y_true - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(y_true, y_pred, alpha=0.55)
        low, high = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax1.plot([low, high], [low, high], "--", color="black")
        ax1.set(xlabel="True", ylabel="Predicted", title="OOF true vs predicted")
        ax2.scatter(y_pred, residual, alpha=0.55)
        ax2.axhline(0, color="black", linestyle="--")
        ax2.set(xlabel="Predicted", ylabel="True − predicted", title="Residuals")
    fig.tight_layout()
    fig.savefig(output_dir / "validation.png", dpi=140)
    plt.close(fig)


def save_cv_scores(fold_scores, config):
    if not bool(config.visualization.save_cv_scores) or not fold_scores:
        return
    labels, scores = zip(*fold_scores)
    values = np.asarray(scores, dtype=float)
    mean, std = values.mean(), values.std()
    fig, ax = plt.subplots(figsize=(max(7, len(scores) * 1.2), 5))
    bars = ax.bar([str(f) for f in labels], values)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.axhline(mean, color="darkred", linestyle="--", label=f"Mean ± std: {mean:.4f} ± {std:.4f}")
    ax.set(xlabel="Fold", ylabel=str(config.metric.name), title="CV scores")
    ax.legend()
    fig.tight_layout()
    path = Path(config.paths.path_to_plots) / "cv_scores.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _get_transformed_data(pipeline, X):
    preprocessor = pipeline.named_steps["preprocessing"]
    transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    return transformed, feature_names


def save_feature_importance(pipeline, X, config) -> None:
    """Save model coefficients or feature_importances_ when the model exposes them."""
    if not config.visualization.save_feature_importance:
        return

    model = pipeline.named_steps["model"]
    _, feature_names = _get_transformed_data(pipeline, X)

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importance = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    else:
        return

    if importance.ndim != 1 or len(importance) != len(feature_names):
        return  # composite estimator has no directly aligned importances
    top_n = min(25, len(feature_names))
    indices = np.argsort(importance)[-top_n:]

    plt.figure(figsize=(9, max(5, top_n * 0.28)))
    plt.barh(np.asarray(feature_names)[indices], importance[indices])
    plt.xlabel("Importance")
    from estimator_strategy import get_estimator_label
    plt.title(f"Feature importance: {get_estimator_label(config)}")
    plt.tight_layout()

    path = Path(config.paths.path_to_plots) / "feature_importance.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


def save_shap_summary(pipeline, X, config) -> None:
    """Optional SHAP plot. Kept as a model-understanding tool, but disabled by default."""
    if not config.visualization.show_shap:
        return

    try:
        import shap
    except ImportError:
        print("SHAP is enabled, but the 'shap' package is not installed")
        return

    sample_size = min(int(config.visualization.shap_max_samples), len(X))
    X_sample = X.sample(sample_size, random_state=int(config.general.seed))
    transformed, feature_names = _get_transformed_data(pipeline, X_sample)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    model = pipeline.named_steps["model"]

    try:
        explainer = shap.Explainer(model, transformed, feature_names=feature_names)
        shap_values = explainer(transformed)
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.tight_layout()
        path = Path(config.paths.path_to_plots) / "shap_summary.png"
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close()
    except Exception as error:
        print(f"SHAP visualization skipped: {error}")
