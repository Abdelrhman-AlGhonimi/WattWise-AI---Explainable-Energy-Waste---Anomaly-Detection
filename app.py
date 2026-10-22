from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.wattwise.config import ROOT, RAW_DATA, ARTIFACTS, REPORTS
from src.wattwise.data_loader import load_data
from src.wattwise.preprocessing import clean_data
from src.wattwise.feature_engineering import make_features
from src.wattwise.inference import load_bundle
from src.wattwise.inference import predict_next
from src.wattwise.explainability import local_shap_values
from src.wattwise.model_factory import model_factory

st.set_page_config(
    page_title="WattWise AI",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(":material/bolt: WattWise AI")
st.caption("Energy forecasting · model evaluation · candidate anomaly review")
pages = [
    "Project Overview",
    "Dataset Explorer",
    "Energy Forecasting",
    "Model Comparison",
    "Anomaly Detection",
    "Explainability",
    "System Architecture and Methodology",
    "Testing and Project Information",
]
st.sidebar.title("WattWise AI")
st.sidebar.caption("Advanced Machine Learning · Team project")
page = st.sidebar.selectbox("Navigate to", pages, key="navigation_page")
dark_mode = st.sidebar.toggle(
    "Dark mode",
    key="dark_mode",
    help="Switch the interface between light and dark appearance.",
)
if dark_mode:
    px.defaults.template = "plotly_dark"
    px.defaults.color_discrete_sequence = [
        "#45c2c9",
        "#67a5ff",
        "#f5ad55",
        "#a99af5",
        "#f17b93",
        "#6fc495",
    ]
    st.html("""<style>
      :root { color-scheme: dark; --text-color: #edf3f7; --background-color: #111820; --secondary-background-color: #1b2833; --border-color: #354654; }
      .stApp, [data-testid="stAppViewContainer"] { background: #111820; color: #edf3f7; }
      [data-testid="stHeader"] { background: #111820; }
      [data-testid="stSidebar"] { background: #17232d; }
      [data-testid="stAppViewContainer"] :is(p,span,label,small,strong,em,h1,h2,h3,h4,h5,h6,li,summary) { color: #edf3f7 !important; }
      [data-testid="stSidebar"] :is(p,span,label,small,strong,em,h1,h2,h3,h4,h5,h6,li,summary) { color: #edf3f7 !important; }
      [data-testid="stMetric"] { background-color: #1b2833; border: 1px solid #354654; border-radius: 0.65rem; padding: 0.7rem; }
      [data-testid="stMetric"] [data-testid="stMetricLabel"], [data-testid="stMetric"] [data-testid="stMetricValue"], [data-testid="stMetric"] [data-testid="stMetricDelta"] { color: #f4f8fb !important; }
      [data-testid="stVerticalBlockBorderWrapper"] { background-color: #141f28; border-color: #465966; }
      [data-baseweb="input"] input, [data-baseweb="select"] > div, [data-baseweb="select"] span, textarea { background: #1b2833; color: #f4f8fb !important; }
      [data-baseweb="popover"] *, [role="listbox"] *, [role="option"] { color: #172b3a !important; }
      [data-testid="stDataFrame"] { color-scheme: dark; }
      a { color: #72d2d5; }
    </style>""")
else:
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = [
        "#087f8c",
        "#2b6cb0",
        "#d97706",
        "#6554c0",
        "#c2415d",
        "#37805a",
    ]
    st.html("""<style>
      :root { color-scheme: light; }
      .stApp, [data-testid="stAppViewContainer"] { background: #f7fafc; color: #172b3a; }
      [data-testid="stHeader"] { background: #f7fafc; }
      [data-testid="stSidebar"] { background: #edf3f6; }
      [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] { background-color: #ffffff; border-color: #d7e2e9; }
      a { color: #087f8c; }
    </style>""")


@st.cache_data(show_spinner="Loading UCI dataset…")
def get_data():
    cleaned, _ = clean_data(load_data())
    return cleaned


@st.cache_resource
def get_bundle():
    return load_bundle(ARTIFACTS / "wattwise_bundle.joblib")


def artifact_json(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


def predict_bundle(bundle, features):
    return predict_next(bundle, features)


def heading(title, description):
    st.header(title)
    st.caption(description)


def format_r2(value, digits=3):
    return "Undefined" if pd.isna(value) else f"{value:.{digits}f}"


def show_dataframe(frame, **kwargs):
    """Keep data tables readable against either app theme."""
    if dark_mode and isinstance(frame, pd.DataFrame):
        frame = frame.style.set_properties(
            **{
                "background-color": "#18242e",
                "color": "#edf3f7",
                "border-color": "#354654",
            }
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#243542"),
                        ("color", "#f4f8fb"),
                        ("border-color", "#465966"),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(even) td",
                    "props": [("background-color", "#1d2a35")],
                },
            ]
        )
    return st.dataframe(frame, **kwargs)


def show_chart(fig, alt):
    """Give Plotly figures explicit canvas and text colors for the active theme."""
    paper = "#111820" if dark_mode else "#f7fafc"
    plot = "#111820" if dark_mode else "#ffffff"
    text_color = "#edf3f7" if dark_mode else "#172b3a"
    grid = "#354654" if dark_mode else "#d7e2e9"
    fig.update_layout(paper_bgcolor=paper, plot_bgcolor=plot, font_color=text_color)
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)
    return st.plotly_chart(fig, width="stretch", theme=None, alt=alt)


def artifact_run_problem(manifest, metrics, config, predictions, comparison, bundle):
    """Return a readable error if dashboard artifacts were mixed across runs."""
    if not manifest:
        return "Artifact manifest is missing; rerun `python -m wattwise.train` to validate a consistent model/report set."
    expected = manifest.get("run_id")
    missing_files = []
    for filename in manifest.get("files", []):
        artifact_path = (
            ARTIFACTS / filename
            if filename == "wattwise_bundle.joblib"
            else REPORTS / filename
            if filename == "model_comparison.csv"
            else REPORTS / "metrics" / filename
        )
        if not artifact_path.exists():
            missing_files.append(filename)
    if missing_files:
        return (
            "The training manifest lists missing artifacts: "
            + ", ".join(missing_files)
            + ". Rerun training to regenerate them."
        )
    sources = {
        "final metrics": metrics,
        "training config": config,
        "test predictions": predictions,
        "model comparison": comparison,
        "model bundle": bundle,
    }
    mismatched = [
        name
        for name, value in sources.items()
        if not value or value.get("run_id") != expected
    ]
    if mismatched:
        return (
            "Artifacts do not belong to the same training run: "
            + ", ".join(mismatched)
            + ". Rerun training to regenerate them together."
        )
    fingerprints = {value.get("dataset_fingerprint") for value in sources.values()}
    if len(fingerprints) != 1 or None in fingerprints:
        return "Artifact dataset fingerprints do not match; rerun training before interpreting results."
    if metrics.get("selected_model") != bundle.get("selected_model") or config.get(
        "selected_model"
    ) != bundle.get("selected_model"):
        return "The saved model bundle and evaluation reports disagree about the selected model. Rerun training before interpreting results."
    if predictions.get("selected_model") != bundle.get("selected_model"):
        return "Saved test predictions were generated by a different model than the loaded bundle. Rerun training before interpreting results."
    if comparison.get("selected_models") != [bundle.get("selected_model")]:
        return "The test comparison does not mark exactly the loaded model as selected. Rerun training before interpreting results."
    metadata_artifacts = {
        "best_hyperparameters.json": REPORTS / "metrics" / "best_hyperparameters.json",
        "anomaly_evaluation.json": REPORTS / "metrics" / "anomaly_evaluation.json",
        "residual_diagnostics.json": REPORTS / "metrics" / "residual_diagnostics.json",
        "selected_features.json": REPORTS / "metrics" / "selected_features.json",
    }
    for label, path in metadata_artifacts.items():
        metadata = artifact_json(path)
        if not metadata or metadata.get("run_id") != expected:
            return f"{label} belongs to a different or incomplete training run. Rerun training before interpreting it."
    for label, path in {
        "feature_ablation.csv": REPORTS / "metrics" / "feature_ablation.csv",
        "feature_importance.csv": REPORTS / "metrics" / "feature_importance.csv",
    }.items():
        frame = pd.read_csv(path, nrows=1)
        if frame.empty or "run_id" not in frame or frame.iloc[0]["run_id"] != expected:
            return f"{label} belongs to a different or incomplete training run. Rerun training before interpreting it."
    return None


try:
    data = get_data()
    data_error = None
except Exception as e:
    data = None
    data_error = str(e)
try:
    bundle = get_bundle()
    bundle_error = None
except Exception as e:
    bundle = None
    bundle_error = str(e)
run_manifest = artifact_json(REPORTS / "metrics" / "artifact_manifest.json")
saved_metrics = artifact_json(REPORTS / "metrics" / "final_metrics.json") or {}
saved_config = artifact_json(REPORTS / "metrics" / "training_config.json") or {}
saved_prediction_header = (
    pd.read_csv(REPORTS / "metrics" / "test_predictions.csv", nrows=1).iloc[0].to_dict()
    if (REPORTS / "metrics" / "test_predictions.csv").exists()
    else None
)
saved_comparison = None
if (REPORTS / "model_comparison.csv").exists():
    comparison_meta = pd.read_csv(REPORTS / "model_comparison.csv")
    selected_rows = (
        comparison_meta[
            (comparison_meta["split"] == "test")
            & comparison_meta["selected"].fillna(False)
        ]
        if "selected" in comparison_meta
        else comparison_meta.iloc[0:0]
    )
    saved_comparison = {
        "run_id": comparison_meta.iloc[0].get("run_id"),
        "dataset_fingerprint": comparison_meta.iloc[0].get("dataset_fingerprint"),
        "selected_models": selected_rows["model"].tolist(),
    }
bundle_metadata = {
    key: bundle[key]
    for key in ("run_id", "dataset_fingerprint", "selected_model")
    if bundle and key in bundle
}
run_problem = (
    artifact_run_problem(
        run_manifest,
        saved_metrics,
        saved_config,
        saved_prediction_header,
        saved_comparison,
        bundle_metadata,
    )
    if bundle
    else "Model bundle unavailable."
)

if page == "Project Overview":
    heading(
        "Project overview",
        "A short-horizon energy forecast with explainable, human-reviewed anomaly signals.",
    )
    st.markdown(
        "Forecast household appliance consumption at the next observed 10-minute interval. The system highlights unusual deviations; it does not diagnose faults or confirm waste."
    )
    st.subheader("Project team")
    st.table(
        pd.DataFrame(
            [
                {
                    "Team member": "Abdelrhman Mohamed Ahmed Al Ghonimi",
                    "ID": "223101228",
                },
                {"Team member": "Denar Reda Roushdy Elsayed", "ID": "223101171"},
                {"Team member": "Mariem Abd-Elhamied Abo-Shaeshaa", "ID": "223101341"},
                {"Team member": "Shrouk Ashraf Seria", "ID": "223101203"},
            ]
        )
    )
    st.caption("Course: Advanced Machine Learning")
    cols = st.columns(4)
    if data is not None:
        cols[0].metric("Observations", f"{len(data):,}")
        cols[1].metric("Raw fields", len(data.columns))
        cols[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
        cols[3].metric("Period", f"{data.date.min():%b %Y} – {data.date.max():%b %Y}")
    metrics = artifact_json(REPORTS / "metrics" / "final_metrics.json")
    if metrics:
        st.subheader("Selected model · untouched test period")
        with st.container(border=True):
            st.write(f"Chosen using validation MAE: **{metrics['selected_model']}**")
            cols = st.columns(4)
            cols[0].metric("MAE", f"{metrics['test']['MAE']:.2f} Wh")
            cols[1].metric("RMSE", f"{metrics['test']['RMSE']:.2f} Wh")
            cols[2].metric("R²", format_r2(metrics["test"]["R2"]))
            cols[3].metric("Features", metrics["feature_count"])
            st.caption(
                "Regression models do not use classification accuracy. MAE, RMSE and R² are the primary measures; percentage error can be misleading when actual use is near zero."
            )
    else:
        st.info(
            "Train models with `python -m wattwise.train` to populate validated results.",
            title="Evaluation artifacts not found",
        )
    st.subheader("Workflow")
    st.graphviz_chart(
        "digraph { UCI -> Raw -> Validation -> Features -> Models -> Evaluation -> Dashboard; Models -> Forecast; Forecast -> Residuals -> Review; }"
    )
elif page == "Dataset Explorer":
    st.header("Dataset explorer")
    if data is None:
        st.error(data_error)
    else:
        st.write(f"{data.shape[0]:,} rows × {data.shape[1]} columns")
        cols = st.multiselect(
            "Columns", data.columns.tolist(), default=["date", "Appliances", "lights"]
        )
        show_dataframe(data[cols].head(500), hide_index=True)
        c1, c2 = st.columns(2)
        c1.plotly_chart(
            px.histogram(
                data,
                x="Appliances",
                nbins=50,
                title="Appliances distribution",
                labels={"Appliances": "Appliances (Wh per 10 min)"},
            ),
            width="stretch",
            theme=None,
        )
        if "date" in data:
            c2.plotly_chart(
                px.line(
                    data.iloc[::20],
                    x="date",
                    y="Appliances",
                    title="Consumption time series (every 20th row)",
                ),
                width="stretch",
                theme=None,
            )
        st.subheader("Data quality")
        quality = pd.DataFrame(
            {"missing": data.isna().sum(), "missing_percent": 100 * data.isna().mean()}
        ).query("missing > 0")
        if quality.empty:
            st.success("No missing values in the acquired dataset.")
        else:
            show_dataframe(quality)
        numeric = data.select_dtypes("number")
        show_chart(
            px.imshow(
                numeric.corr(),
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                title="Numeric feature correlations",
            ),
            alt="Heatmap of correlations between numeric measurements",
        )
elif page == "Energy Forecasting":
    heading(
        "Energy forecasting",
        "Select a historical feature row and inspect the model’s next-interval forecast.",
    )
    if bundle is None:
        st.warning(
            f"Model bundle unavailable: {bundle_error}. Run `python -m wattwise.train`.",
            title="Model not ready",
        )
    elif data is None:
        st.error(data_error)
    else:
        X, y, dates = make_features(data)
        idx = st.slider(
            "Historical feature row",
            min_value=0,
            max_value=len(X) - 1,
            value=len(X) - 1,
            help="The target is the next observed Appliances value.",
        )
        feats = X.iloc[[idx]]
        try:
            if run_problem:
                st.warning(run_problem, title="Artifact consistency check")
            pred = predict_bundle(bundle, feats)
            actual = y.iloc[idx]
            with st.container(border=True):
                st.caption(
                    f"Forecast origin: {dates.iloc[idx]} · target: next observed interval"
                )
                a, b, c = st.columns(3)
                a.metric("Predicted consumption", f"{pred[0]:.1f} Wh")
                b.metric("Observed next value", f"{actual:.1f} Wh")
                c.metric("Historical residual", f"{actual - pred[0]:+.1f} Wh")
            st.subheader("Input feature snapshot")
            show_dataframe(
                feats.T.rename(columns={feats.index[0]: "Value"}), height=420
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
elif page == "Model Comparison":
    heading(
        "Model comparison",
        "Review all saved regression scores, switch between validation and test periods, and inspect model details.",
    )
    path = REPORTS / "model_comparison.csv"
    if path.exists():
        if run_problem:
            st.error(run_problem, title="Inconsistent model artifacts")
        tab = pd.read_csv(path)
        split = st.radio(
            "Evaluation period",
            tab.split.unique().tolist(),
            index=list(tab.split.unique()).index("test")
            if "test" in tab.split.unique()
            else 0,
            horizontal=True,
            key="comparison_period",
        )
        view = tab[tab.split == split].copy()
        summary = artifact_json(REPORTS / "metrics" / "final_metrics.json") or {}
        selected = summary.get("selected_model")
        selection = summary.get("selection", {})
        if selection:
            st.caption(
                f"Selected before testing by {selection.get('metric')}: **{selection.get('selected_model')}** · validation MAE {selection.get('score'):.3f} Wh. Test scores below are final held-out measurements, not the selection criterion."
            )
        if split == "test" and selected:
            selected_row = view[view.model == selected]
            if not selected_row.empty:
                row = selected_row.iloc[0]
                cols = st.columns(4)
                cols[0].metric("Selected model", selected)
                cols[1].metric("MAE", f"{row.MAE:.2f} Wh")
                cols[2].metric("RMSE", f"{row.RMSE:.2f} Wh")
                cols[3].metric("R²", format_r2(row.R2))
                retrospective = view.sort_values(["MAE", "RMSE", "model"]).iloc[0]
                st.caption(
                    f"Retrospective test MAE leader: {retrospective.model} ({retrospective.MAE:.3f} Wh). This held-out comparison does not change the validation-selected model."
                )
        st.markdown("#### All model results")
        view = view.sort_values("MAE").reset_index(drop=True)
        if split == "training_cv":
            st.caption(
                "Training-fold estimates only. Tuned HistGradientBoosting uses nested temporal CV; voting/stacking are validation-selected ensembles, so no biased ensemble CV score is shown for them. Standard deviations use population spread across folds."
            )
            display = (
                view[
                    [
                        "model",
                        "CV_MAE",
                        "CV_MAE_std",
                        "CV_RMSE",
                        "CV_RMSE_std",
                        "CV_R2",
                        "CV_R2_std",
                        "n_folds",
                        "mean_fold_n_obs",
                    ]
                ]
                .rename(
                    columns={
                        "model": "Model",
                        "CV_MAE": "Mean CV MAE (Wh)",
                        "CV_MAE_std": "CV MAE std (Wh)",
                        "CV_RMSE": "Mean CV RMSE (Wh)",
                        "CV_RMSE_std": "CV RMSE std (Wh)",
                        "CV_R2": "Mean CV R²",
                        "CV_R2_std": "CV R² std",
                        "n_folds": "Folds",
                        "mean_fold_n_obs": "Mean validation rows",
                    }
                )
                .round(3)
            )
        else:
            display = (
                view[["model", "MAE", "RMSE", "R2", "n_obs"]]
                .rename(
                    columns={
                        "model": "Model",
                        "MAE": "MAE (Wh / 10 min)",
                        "RMSE": "RMSE (Wh / 10 min)",
                        "R2": "R²",
                        "n_obs": "Observations",
                    }
                )
                .round({"MAE (Wh / 10 min)": 3, "RMSE (Wh / 10 min)": 3, "R²": 4})
            )
            if "selected" in view and view["selected"].fillna(False).any():
                display["Selection"] = view["selected"].map(
                    lambda value: "✓ Selected" if value else ""
                )
        show_dataframe(display, hide_index=True)
        metric_name = st.selectbox(
            "Metric to compare", ["MAE", "RMSE", "R2"], key="model_metric"
        )
        comparison_fig = px.bar(
            view.sort_values(metric_name, ascending=(metric_name != "R2")),
            x="model",
            y=metric_name,
            color="selected" if "selected" in view else None,
            title=f"{metric_name} · {split} period",
            labels={
                "MAE": "MAE (Wh / 10 min)",
                "RMSE": "RMSE (Wh / 10 min)",
                "R2": "R²",
            },
        )
        show_chart(
            comparison_fig,
            alt=f"Model comparison by {metric_name} for the {split} period",
        )
        model_names = view.model.tolist()
        chosen = st.selectbox(
            "Inspect model details",
            model_names,
            index=model_names.index(selected) if selected in model_names else 0,
        )
        detail = view[view.model == chosen]
        if not detail.empty:
            with st.container(border=True):
                st.markdown(f"**{chosen}**")
                st.write(
                    "MAE and RMSE are error in Wh per 10-minute interval. R² is variance explained relative to a mean prediction and can be negative."
                )
                if split == "training_cv":
                    chosen_row = detail.iloc[0]
                    cards = st.columns(4)
                    cards[0].metric("Mean CV MAE", f"{chosen_row.CV_MAE:.3f} Wh")
                    cards[1].metric(
                        "CV MAE variation", f"±{chosen_row.CV_MAE_std:.3f} Wh"
                    )
                    cards[2].metric("Mean CV RMSE", f"{chosen_row.CV_RMSE:.3f} Wh")
                    cards[3].metric("Mean CV R²", format_r2(chosen_row.CV_R2, 4))
                    st.caption(
                        f"CV R² fold standard deviation: {format_r2(chosen_row.CV_R2_std, 4)}"
                    )
                else:
                    chosen_row = detail.iloc[0]
                    cards = st.columns(3)
                    cards[0].metric("MAE", f"{chosen_row.MAE:.3f} Wh / 10 min")
                    cards[1].metric("RMSE", f"{chosen_row.RMSE:.3f} Wh / 10 min")
                    cards[2].metric("R²", format_r2(chosen_row.R2, 4))
                    st.caption(
                        f"Evaluated on {int(chosen_row.n_obs):,} chronological {split} observations; target is next observed row in Wh per 10-minute interval."
                    )
        with st.expander("Model formulation and parameters"):
            catalog = model_factory()
            if chosen in catalog:
                st.write(f"Estimator class: `{type(catalog[chosen]).__name__}`")
                st.json(catalog[chosen].get_params(deep=False))
            elif chosen == "Persistence":
                st.write(
                    "Forecast = current Appliances reading at time t. This baseline has no fitted parameters."
                )
            elif chosen == "Tuned HistGradientBoosting":
                st.write(
                    "Histogram-based boosting with absolute-error loss. The selected final parameters are trained on train + validation; tuned-model CV estimates use nested expanding folds, with tuning confined to each outer training block."
                )
                st.json(
                    (
                        artifact_json(REPORTS / "metrics" / "best_hyperparameters.json")
                        or {}
                    )
                    .get("optimization", {})
                    .get("best_params", {})
                )
            elif (
                chosen == "Temporal Stacking"
                and bundle
                and isinstance(bundle.get("model"), dict)
            ):
                stack = bundle["model"].get("stack", {})
                st.write(
                    "Walk-forward out-of-fold base predictions are combined by a Ridge meta-learner."
                )
                st.json(
                    {
                        "base_estimators": {
                            name: {
                                "class": type(model).__name__,
                                "parameters": model.get_params(deep=False),
                            }
                            for name, model in stack.get("base_models", {}).items()
                        },
                        "meta_estimator": type(stack.get("meta_model")).__name__
                        if stack.get("meta_model")
                        else None,
                        "meta_parameters": stack["meta_model"].get_params(deep=False)
                        if stack.get("meta_model")
                        else {},
                    }
                )
            elif chosen == "Voting (top 3)" and bundle:
                st.write(
                    "Equal-weight average of the three best models selected by validation MAE."
                )
                st.json(
                    bundle.get("model", {}).get(
                        "weights",
                        summary.get("selection", {}).get("voting_weights", {}),
                    )
                )
            else:
                st.write(
                    "See the training configuration and optimization report for this estimator's saved setup."
                )
        pred_path = REPORTS / "metrics" / "test_predictions.csv"
        if split == "test" and chosen == selected and pred_path.exists():
            selected_predictions = pd.read_csv(pred_path)
            actual = selected_predictions["observed"].abs()
            valid = actual > 0
            wape = (
                100
                * selected_predictions.loc[valid, "residual"].abs().sum()
                / actual.loc[valid].sum()
            )
            st.metric(
                "Selected-model WAPE",
                f"{wape:.2f}%",
                help="Sum of absolute test errors divided by sum of absolute observed use. This percentage metric is only available for the selected model's saved predictions.",
            )
            st.caption(
                "Regression has no classification accuracy score. WAPE is a relative error summary, not the percentage of predictions that are correct."
            )
        tuned = artifact_json(REPORTS / "metrics" / "best_hyperparameters.json")
        if tuned:
            with st.expander("Hyperparameter optimization details"):
                st.write(
                    f"{tuned['tuned_model']} · {tuned['optimization'].get('backend', 'unknown')} · best inner training-fold tuning objective: {tuned['optimization'].get('best_cv_mae', float('nan')):.3f} Wh. This is a tuning score; use nested CV for performance estimates."
                )
                st.json(tuned["optimization"].get("best_params", {}))
                history = tuned["optimization"].get("history", [])
                if history:
                    show_chart(
                        px.line(
                            pd.DataFrame(history),
                            x="number",
                            y="value",
                            markers=True,
                            title="Optuna objective by trial (training folds)",
                        ),
                        alt="Optuna objective score by trial",
                    )
        ablation_path = REPORTS / "metrics" / "feature_ablation.csv"
        if ablation_path.exists():
            st.subheader("Feature ablation · validation only")
            show_dataframe(pd.read_csv(ablation_path).round(4), hide_index=True)
        pred_path = REPORTS / "metrics" / "test_predictions.csv"
        if split == "test" and pred_path.exists():
            p = pd.read_csv(pred_path)
            show_chart(
                px.line(
                    p,
                    x="target_date",
                    y=["observed", "predicted"],
                    title="Observed and selected-model predicted consumption",
                    labels={
                        "value": "Wh / 10 min",
                        "variable": "Series",
                        "target_date": "Target timestamp",
                    },
                ),
                alt="Observed versus predicted consumption across the temporal test set",
            )
            diagnostic_path = REPORTS / "metrics" / "residual_diagnostics.json"
            diagnostic = artifact_json(diagnostic_path)
            if diagnostic:
                with st.expander(
                    "Residual diagnostics · held-out test", expanded=False
                ):
                    q = pd.DataFrame(
                        [
                            {
                                "Quantile": label,
                                "Absolute error (Wh)": diagnostic["quantiles"][key],
                            }
                            for label, key in [
                                ("Median", "absolute_error_p50"),
                                ("90th percentile", "absolute_error_p90"),
                                ("95th percentile", "absolute_error_p95"),
                                ("99th percentile", "absolute_error_p99"),
                            ]
                        ]
                    )
                    st.write(
                        "The large RMSE reflects the tail of the absolute-error distribution. These errors are retained; no test observations were removed."
                    )
                    show_dataframe(q.round(2), hide_index=True)
                    show_chart(
                        px.histogram(
                            p,
                            x="residual",
                            nbins=40,
                            title="Residual distribution (observed − predicted)",
                            labels={"residual": "Residual (Wh)"},
                        ),
                        alt="Distribution of held-out residuals",
                    )
                    actual_predicted = px.scatter(
                        p,
                        x="observed",
                        y="predicted",
                        title="Actual versus predicted consumption",
                        labels={
                            "observed": "Observed (Wh)",
                            "predicted": "Predicted (Wh)",
                        },
                    )
                    limits = [
                        min(p.observed.min(), p.predicted.min()),
                        max(p.observed.max(), p.predicted.max()),
                    ]
                    actual_predicted.add_shape(
                        type="line",
                        x0=limits[0],
                        y0=limits[0],
                        x1=limits[1],
                        y1=limits[1],
                        line={"dash": "dash", "color": "#92a4b2"},
                    )
                    show_chart(
                        actual_predicted,
                        alt="Actual versus predicted test consumption with ideal prediction line",
                    )
                    residual_fig = px.scatter(
                        p,
                        x="predicted",
                        y="residual",
                        title="Residuals versus prediction",
                        labels={
                            "predicted": "Predicted (Wh)",
                            "residual": "Residual (Wh)",
                        },
                    )
                    residual_fig.add_hline(y=0, line_dash="dash")
                    show_chart(
                        residual_fig, alt="Held-out residuals versus predictions"
                    )
                    show_chart(
                        px.line(
                            p,
                            x="target_date",
                            y="residual",
                            title="Residuals over time",
                            labels={
                                "target_date": "Target timestamp",
                                "residual": "Residual (Wh)",
                            },
                        ),
                        alt="Held-out residuals over target time",
                    )
                    st.markdown("**Error by time block**")
                    show_dataframe(
                        pd.DataFrame(diagnostic["by_time_block"]), hide_index=True
                    )
                    st.markdown("**Error by observed consumption range**")
                    show_dataframe(
                        pd.DataFrame(diagnostic["by_target_range"]), hide_index=True
                    )
                    st.markdown("**Largest absolute errors**")
                    show_dataframe(
                        pd.DataFrame(diagnostic["largest_errors"]), hide_index=True
                    )
    else:
        st.info("No evaluation artifact found. Run training first.")
elif page == "Anomaly Detection":
    heading(
        "Candidate unusual events",
        "Choose a detector and inspect the observations that merit human review.",
    )
    path = REPORTS / "metrics" / "test_predictions.csv"
    metrics = artifact_json(REPORTS / "metrics" / "final_metrics.json")
    if path.exists():
        p = pd.read_csv(path)
        method = st.radio(
            "Detection method",
            ["Forecast residual", "Isolation Forest"],
            horizontal=True,
            key="anomaly_method",
        )
        flag_col = "anomaly" if method == "Forecast residual" else "isolation_anomaly"
        p["date"] = pd.to_datetime(p["date"])
        date_range = st.date_input(
            "Held-out date range",
            value=(p.date.min().date(), p.date.max().date()),
            min_value=p.date.min().date(),
            max_value=p.date.max().date(),
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            lo, hi = date_range
            p = p[p.date.dt.date.between(lo, hi)]
        flagged = p[p[flag_col]]
        st.metric("Candidate events", len(flagged))
        show_dataframe(flagged, hide_index=True)
        hover = [c for c in ["predicted", "residual", "isolation_score"] if c in p]
        show_chart(
            px.scatter(
                p,
                x="date",
                y="observed",
                color=flag_col,
                hover_data=hover,
                title=f"Observed values flagged by {method}",
            ),
            alt=f"Held-out consumption candidates flagged by {method}",
        )
        if method == "Forecast residual":
            st.caption(
                f"Calibration threshold: absolute validation residual 99th percentile = {metrics['threshold']:.2f} Wh."
            )
        else:
            st.caption(
                "Isolation Forest was fitted on training feature vectors. Its score is a relative outlier ranking, not a probability."
            )
        syn = artifact_json(REPORTS / "metrics" / "anomaly_evaluation.json")
        if syn:
            st.subheader(
                "Synthetic benchmark (separate from real unlabeled observations)"
            )
            st.json(syn)
    else:
        st.info("Train the project to generate held-out residuals.")
elif page == "Explainability":
    heading(
        "Explainability",
        "Explore global model reliance and calculate a local SHAP explanation for one forecast.",
    )
    path = REPORTS / "metrics" / "feature_importance.csv"
    if path.exists():
        importance_data = pd.read_csv(path)
        if (
            bundle
            and "selected_model" in importance_data
            and not importance_data.empty
            and importance_data.iloc[0]["selected_model"]
            != bundle.get("selected_model")
        ):
            st.warning(
                "Feature-importance results were produced for a different model than the loaded bundle. Retrain to refresh artifacts."
            )
        imp = importance_data.head(20)
        show_chart(
            px.bar(
                imp.sort_values("importance"),
                x="importance",
                y="feature",
                orientation="h",
                title="Permutation importance · training reference",
            ),
            alt="Permutation feature importance for the selected forecasting model on training data",
        )
        st.caption(
            "Bars show the increase in training MAE after permuting a feature for the selected model. Importance is model reliance, not causation or independent performance evidence."
        )
        if data is not None and bundle is not None:
            X, y, dates = make_features(data)
            idx = st.slider(
                "Held-out explanation row",
                0,
                len(X) - 1,
                max(0, len(X) - 1),
                key="explain_idx",
            )
            st.caption(f"Input time: {dates.iloc[idx]}")
            if st.button("Calculate local SHAP explanation"):

                def predict_fn(z):
                    return predict_bundle(bundle, z)

                explanation, error = local_shap_values(
                    predict_fn, X.iloc[: min(5000, len(X))], X.iloc[[idx]]
                )
                if error:
                    st.info(
                        error
                        + " Showing global feature importance as the available fallback."
                    )
                elif explanation is not None:
                    show_chart(
                        px.bar(
                            explanation.head(15).sort_values("contribution"),
                            x="contribution",
                            y="feature",
                            orientation="h",
                            title="Local SHAP contribution to this forecast",
                        ),
                        alt="Local SHAP feature contributions for the selected forecast",
                    )
                    st.caption(
                        "Positive contributions raise the model output relative to its background reference; negative contributions lower it. These are model attributions, not causal effects."
                    )
    else:
        st.info(
            "Feature explanation artifact is created during training. Permutation importance is the fallback when native tree importance is unavailable."
        )
elif page == "System Architecture and Methodology":
    st.header("Architecture and methodology")
    st.markdown(
        "```mermaid\nflowchart LR\nU[UCI archive] --> R[Raw CSV] --> V[Validation and cleaning] --> F[Past-only features]\nF --> S[Chronological split]\nS --> T[Train and tune on history]\nT --> E[Validation selection]\nE --> H[Held-out test]\nH --> D[Streamlit: forecast, residual, explanation]\n```\n"
    )
    for path in sorted((ROOT / "docs" / "diagrams").glob("*.mmd")):
        with st.expander(path.stem):
            st.code(path.read_text(encoding="utf-8"), language="mermaid")
elif page == "Testing and Project Information":
    heading(
        "Project tests and information",
        "Run the automated test suite on demand and check the readiness of data, model, and report artifacts.",
    )
    st.write(
        "Run tests executes `python -m pytest -q` inside the project folder. It does not retrain the models."
    )
    checks = [
        {
            "check": "UCI raw dataset",
            "status": "Ready"
            if RAW_DATA.exists() and data is not None
            else "Missing or invalid",
            "detail": f"{len(data):,} rows · {len(data.columns)} columns"
            if data is not None
            else str(data_error),
        },
        {
            "check": "Saved model bundle",
            "status": "Ready" if bundle is not None else "Missing or failed to load",
            "detail": str(ARTIFACTS / "wattwise_bundle.joblib"),
        },
        {
            "check": "Model comparison",
            "status": "Ready"
            if (REPORTS / "model_comparison.csv").exists()
            else "Missing",
            "detail": "Validation and test scores",
        },
        {
            "check": "Artifact consistency",
            "status": "Ready" if not run_problem else "Check required",
            "detail": run_problem
            or f"Run {run_manifest.get('run_id', '')[:8]} · model {saved_metrics.get('selected_model', 'unknown')}",
        },
        {
            "check": "Final metrics",
            "status": "Ready"
            if (REPORTS / "metrics" / "final_metrics.json").exists()
            else "Missing",
            "detail": "Selected model and held-out test metrics",
        },
        {
            "check": "Anomaly report",
            "status": "Ready"
            if (REPORTS / "metrics" / "anomaly_evaluation.json").exists()
            else "Missing",
            "detail": "Synthetic benchmark and detector metrics",
        },
    ]
    show_dataframe(pd.DataFrame(checks), hide_index=True)
    if st.button("Run automated project tests", type="primary", key="run_pytest"):
        try:
            with st.spinner("Running pytest. This may take a few seconds…"):
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False,
                )
            st.session_state["project_test_result"] = {
                "returncode": result.returncode,
                "output": (result.stdout + "\n" + result.stderr).strip(),
            }
        except Exception as exc:
            st.session_state["project_test_result"] = {
                "returncode": 1,
                "output": str(exc),
            }
    test_result = st.session_state.get("project_test_result")
    if test_result:
        if test_result["returncode"] == 0:
            st.success("Project test suite passed.", icon=":material/check_circle:")
        else:
            st.error("Project test suite reported a failure.", icon=":material/error:")
        st.code(test_result["output"] or "No test output", language="text")
    with st.expander("Reproducibility and selected model"):
        config = artifact_json(REPORTS / "metrics" / "training_config.json") or {}
        if bundle:
            st.json(
                {
                    "selected_model": bundle.get("selected_model"),
                    "version": bundle.get("version"),
                    "seed": bundle.get("seed"),
                    "training_period": bundle.get("training_period"),
                    "test_period": bundle.get("test_period"),
                    "base_models": bundle.get("bestnames"),
                }
            )
        if config:
            st.json(config)
    with st.expander("Project documentation"):
        for f in [
            "README.md",
            "docs/PROJECT_DOCUMENTATION.md",
            "docs/METHODOLOGY.md",
            "docs/DATA_DICTIONARY.md",
            "docs/MODEL_CARD.md",
            "docs/LIMITATIONS_AND_ETHICS.md",
        ]:
            st.markdown(f"- [{f}]({f})")
