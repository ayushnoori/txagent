#!/usr/bin/env python

"""
Generates two forest plots from odds ratio analysis results: one for unadjusted
and one for confounder-adjusted odds ratios. It uses Rich for clear console output.

The script reads 'unadjusted_OR_results.csv' and 'adjusted_OR_results.csv' files
from result folders, extracts the relevant data, summarizes them in console tables,
and saves the forest plots as both PNG and PDF files.

Run with:
    uv run python code/ae_prevalence/forest_plot.py
"""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Set font to Arial
plt.rcParams['font.sans-serif'] = 'Arial'

# --- Script Configuration ---

# Initialize pretty printing console
console = Console()

# Define the root directory where results are stored.
# Assumes the script is run from the same root as the 'results' folder.
RESULTS_DIR = Path("results/ae_prevalence")

# Define the output directory for the final plot.
PLOT_OUTPUT_DIR = RESULTS_DIR / "plots"

# --- Define the list of results to include in the forest plot ---
# Each dictionary specifies one analysis to find and plot.
PLOTS_TO_GENERATE: List[Dict[str, Any]] = [
    {
        "disease": "hypertension",
        "comorbidity": "gout",
        "drug_group_name": "diuretic",
        "adverse_event": "squamous cell carcinoma",
    },
    {
        "disease": "diabetes",
        "comorbidity": "ischemic heart disease",
        "drug_group_name": "DPP-4 inhibitor",
        "adverse_event": "hepatocellular carcinoma",
    },
    {
        "disease": "hypertension",
        "comorbidity": "gout",
        "drug_group_name": "beta-blocker",
        "adverse_event": "acute kidney failure",
    },
    {
        "disease": "hypertension",
        "comorbidity": "gout",
        "drug_group_name": "beta-blocker",
        "adverse_event": "hyperkalemia",
    },
    {
        "disease": "bronchial_asthma",
        "comorbidity": "ischemic heart disease",
        "drug_group_name": "long-acting beta-2 agonist",
        "adverse_event": "stroke",
    },
    {
        "disease": "hyperlipidemia",
        "comorbidity": "hypothyroidism",
        "drug_group_name": "statin",
        "adverse_event": "liver failure",
    },
    {
        "disease": "diabetes",
        "comorbidity": "chronic kidney disease",
        "drugs": ["metformin"],
        "adverse_event": "respiratory failure",
    }
]

# Define positive control analyses
POSITIVE_CONTROLS: List[Dict[str, Any]] = [
    {
        "disease": "hypertension",
        "comorbidity": "chronic kidney disease",
        "drug_group_name": "ACE inhibitor",
        "adverse_event": "hyperkalemia",
    },
    {
        "disease": "diabetes",
        "comorbidity": "chronic kidney disease",
        "drug_group_name": "metformin",
        "adverse_event": "acidosis",
    },
    {
        "disease": "hyperlipidemia",
        "comorbidity": "hypothyroidism",
        "drug_group_name": "statin",
        "adverse_event": "rhabdomyolysis",
    },
    {
        "disease": "bronchial_asthma",
        "comorbidity": "ischemic heart disease",
        "drug_group_name": "beta-2 agonist",
        "adverse_event": "angina",
    }
]

# Define negative control analyses
NEGATIVE_CONTROLS: List[Dict[str, Any]] = [
    # {
    #     "disease": "hypertension",
    #     "comorbidity": "chronic kidney disease",
    #     "drug_group_name": "ACE inhibitor",
    #     "adverse_event": "influenza",
    # },
    # {
    #     "disease": "diabetes",
    #     "comorbidity": "chronic kidney disease",
    #     "drug_group_name": "SGLT-2 inhibitor",
    #     "adverse_event": "heart failure",
    # },
    # {
    #     "disease": "diabetes",
    #     "comorbidity": "ischemic heart disease",
    #     "drug_group_name": "GLP-1 receptor agonist",
    #     "adverse_event": "myocardial infarction",
    # },
    # {
    #     "disease": "bipolar disorder",
    #     "comorbidity": "hypothyroidism",
    #     "drug_group_name": "lithium",
    #     "adverse_event": "suicide attempt",
    # }
]

# --- Helper Functions ---

def slugify(text: str) -> str:
    """Converts text to a URL-friendly slug consistent with the analysis script."""
    return "".join(ch.lower() if ch.isalnum() else '_' for ch in text).strip("_")

def build_results_path(config: Dict[str, Any]) -> Path:
    """Builds the expected odds_ratio_results.csv path for a given config."""
    disease_slug = slugify(config["disease"])
    if "drug_group_name" in config:
        drug_slug = slugify(config["drug_group_name"])
    else:
        drug_slug = slugify('_'.join(config["drugs"]))

    dir_name = slugify(f"{config['disease']}-{config['comorbidity']}-{drug_slug}")
    return RESULTS_DIR / disease_slug / dir_name

def summarize_plot_items(plot_data: List[Dict[str, Any]], title: str) -> None:
    """
    Print a Rich table summarizing the entries included in the plot.
    """
    if not plot_data:
        console.print("[bold yellow]No entries collected for plotting.[/bold yellow]")
        return

    table = Table(
        title=title,
        header_style="bold magenta",
        show_lines=False,
    )
    table.add_column("Label", style="yellow")
    table.add_column("OR", justify="right", style="cyan")
    table.add_column("95% CI", justify="right", style="cyan")

    # Sort data by descending OR for the summary table
    sorted_data = sorted(plot_data, key=lambda x: x['odds_ratio'], reverse=True)

    for item in sorted_data:
        or_val = item["odds_ratio"]
        lo = item["ci95_low"]
        hi = item["ci95_high"]
        ci_text = f"{lo:.2f}–{hi:.2f}"
        table.add_row(item["label"], f"{or_val:.2f}", ci_text)

    console.print(table)
    console.print(f"[italic]Total: {len(plot_data)} item(s).[/italic]")


# --- Plotting Function ---

def generate_and_save_plot(plot_data: List[Dict[str, Any]], plot_title: str, output_filename_base: str):
    """Generates and saves a single forest plot with grouped data."""
    if not plot_data:
        console.print(f"\n[bold yellow]No data available for '{plot_title}' plot. Skipping generation.[/bold yellow]")
        return

    console.print(Panel(f"Step 2: Generating {plot_title} Forest Plot", title_align="left", border_style="blue"))
    # FIX: The title argument is now correctly handled by the updated function.
    summarize_plot_items(plot_data, title=f"Included Entries for {plot_title} Forest Plot")

    # Group data by group_type and sort within each group by OR (ascending for plot)
    grouped_data = []
    group_boundaries = []  # Track where groups change for dividing lines
    current_pos = 0
    
    for group_name in ["Main", "Positive Controls", "Negative Controls"]:
        group_items = [item for item in plot_data if item.get("group_type", "Main") == group_name]
        if group_items:
            # Sort items within group by odds ratio (descending, so highest OR at top after inversion)
            group_items_sorted = sorted(group_items, key=lambda x: x['odds_ratio'], reverse=True)
            grouped_data.extend(group_items_sorted)
            current_pos += len(group_items_sorted)
            if current_pos < len(plot_data):  # Don't add boundary after last group
                group_boundaries.append(current_pos - 0.5)  # Place line between groups
    
    if not grouped_data:
        console.print(f"\n[bold yellow]No data available after grouping for '{plot_title}' plot. Skipping generation.[/bold yellow]")
        return
    
    plot_df = pd.DataFrame(grouped_data)
    fig, ax = plt.subplots(figsize=(10, 2 + len(plot_df) * 0.4))
    y_pos = np.arange(len(plot_df))

    # Define color mapping for groups
    color_map = {
        "Main": "black",
        "Positive Controls": "blue",
        "Negative Controls": "red"
    }
    
    # Assign colors to each row based on group_type
    colors = [color_map.get(row.get("group_type", "Main"), "black") for row in grouped_data]

    # Plot error bars and points with group-specific colors
    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = colors[i]
        ax.errorbar(
            x=row["odds_ratio"], y=y_pos[i], 
            xerr=[[row["odds_ratio"] - row["ci95_low"]], [row["ci95_high"] - row["odds_ratio"]]], 
            fmt="none", ecolor=color, capsize=4, elinewidth=1.2, zorder=1
        )
        ax.scatter(x=row["odds_ratio"], y=y_pos[i], marker="s", s=40, color=color, zorder=2)
    
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, zorder=0)

    # Add horizontal dividing lines between groups
    for boundary in group_boundaries:
        ax.axhline(y=boundary, color="gray", linestyle="-", linewidth=1.5, zorder=0, alpha=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=10)
    ax.invert_yaxis()  # Invert so Main group appears at top
    
    x_max = plot_df['ci95_high'].max() * 1.05
    ax.set_xlim(0, x_max)
    
    ax.set_xlabel(f"Odds ratio (95% CI)", fontsize=12)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.grid(axis="x", linestyle=":", color="gray", alpha=0.5)

    ax.text(1.02, 1.0, "OR [95% CI]", transform=ax.transAxes, ha="left", va="bottom", weight="bold", fontsize=10)

    for i, row in plot_df.iterrows():
        or_text = f"{row['odds_ratio']:.2f} ({row['ci95_low']:.2f}–{row['ci95_high']:.2f})"
        ax.text(1.02, y_pos[i], or_text, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=10)

    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path_png = PLOT_OUTPUT_DIR / f"{output_filename_base}.png"
    output_path_pdf = PLOT_OUTPUT_DIR / f"{output_filename_base}.pdf"

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    try:
        plt.savefig(output_path_png, dpi=600, bbox_inches="tight")
        console.print(f"  [green]Saved[/green] PNG → [bold cyan]{output_path_png}[/bold cyan]")
    except Exception as e:
        console.print(f"  [bold red]ERROR:[/] Failed to save PNG: {e}")

    try:
        plt.savefig(output_path_pdf, bbox_inches="tight")
        console.print(f"  [green]Saved[/green] PDF → [bold cyan]{output_path_pdf}[/bold cyan]")
    except Exception as e:
        console.print(f"  [bold red]ERROR:[/] Failed to save PDF: {e}")
    
    plt.close(fig)


# --- Main Script Logic ---

def main():
    """Reads specified odds ratio results, generates forest plots, and saves them."""
    console.print(Panel("[bold magenta]Forest Plot Generator[/bold magenta]", subtitle="Starting...", expand=False))
    console.print(Panel("Step 1: Collecting odds ratio rows", title_align="left", border_style="blue"))

    unadjusted_plot_data: List[Dict[str, Any]] = []
    adjusted_plot_data: List[Dict[str, Any]] = []

    analysis_types = {
        "Unadjusted": {
            "data_list": unadjusted_plot_data,
            "filename": "unadjusted_OR_results.csv",
            "or_col": "odds_ratio"
        },
        "Adjusted": {
            "data_list": adjusted_plot_data,
            "filename": "adjusted_OR_results.csv",
            "or_col": "adjusted_odds_ratio"
        }
    }

    # Define all config sources with their group labels
    config_sources = [
        ("Main", PLOTS_TO_GENERATE),
        ("Positive Controls", POSITIVE_CONTROLS),
        ("Negative Controls", NEGATIVE_CONTROLS),
    ]

    progress_cols = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        TextColumn("OK: {task.completed} of {task.total}")
    ]

    for analysis_type, details in analysis_types.items():
        console.print(f"\n[bold]--- Collecting data for {analysis_type.lower()} analysis ---[/bold]")
        
        # Calculate total configurations across all sources
        total_configs = sum(len(configs) for _, configs in config_sources)
        
        with Progress(*progress_cols, console=console) as progress:
            task = progress.add_task(f"Scanning for {analysis_type.lower()} results...", total=total_configs)

            # Process each config source with its group label
            for group_type, configs in config_sources:
                for config in configs:
                    results_dir = build_results_path(config)
                    results_file = results_dir / details["filename"]

                    if not results_file.exists():
                        progress.update(task, advance=1)
                        continue

                    try:
                        df = pd.read_csv(results_file)
                    except Exception as e:
                        progress.print(f"  [bold red]ERROR:[/] Failed to read [cyan]{results_file}[/cyan] — {e}")
                        progress.update(task, advance=1)
                        continue

                    target_ae = config["adverse_event"].lower().replace("_", " ")
                    target_window = config.get("window", "any_after_index")

                    required_cols = {"adverse_event", "window", details['or_col'], "ci95_low", "ci95_high"}
                    if not required_cols.issubset(df.columns):
                        progress.print(f"  [bold red]ERROR:[/] Missing required columns in [cyan]{results_file.name}[/cyan]. Skipping.")
                        progress.update(task, advance=1)
                        continue

                    row = df[(df["adverse_event"] == target_ae) & (df["window"] == target_window)]

                    if row.empty:
                        progress.update(task, advance=1)
                        continue

                    or_data = row.iloc[0]

                    if "drug_group_name" in config:
                        drug_name_raw = config["drug_group_name"]
                    else:
                        drug_name_raw = ', '.join(config["drugs"])

                    drug_display = (drug_name_raw.lower().replace("_", " ")
                                    .replace("beta-2", "β").replace("dpp-4", "DPP-4"))
                    disease_display = config["disease"].lower().replace("_", " ")
                    comorbidity_display = config["comorbidity"].lower().replace("_", " ")
                    ae_display = str(or_data['adverse_event']).lower()

                    label = (f"{drug_display}\n"
                             f"vs. {ae_display}\n"
                             f"({disease_display} + {comorbidity_display})")

                    try:
                        or_val = float(or_data[details["or_col"]])
                        ci_lo = float(or_data["ci95_low"])
                        ci_hi = float(or_data["ci95_high"])
                    except (ValueError, TypeError):
                        progress.print(f"  [bold red]ERROR:[/] Non-numeric OR/CI values in [cyan]{results_file.name}[/cyan]. Skipping.")
                        progress.update(task, advance=1)
                        continue

                    if not (np.isfinite(or_val) and np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= or_val <= ci_hi):
                        progress.print(f"  [bold yellow]WARNING:[/] Invalid or non-finite OR/CI values in [cyan]{results_file.name}[/cyan]. Skipping.")
                        progress.update(task, advance=1)
                        continue

                    details["data_list"].append({
                        "label": label, "odds_ratio": or_val,
                        "ci95_low": ci_lo, "ci95_high": ci_hi,
                        "group_type": group_type,  # Tag with group type
                    })
                    progress.print(f"  [green]OK[/green] Added ({group_type}): [bold]{drug_display}[/bold] / [italic]{target_ae}[/italic]")
                    progress.update(task, advance=1)

    console.print("\n" + "="*50)
    
    # Generate and save the plots
    generate_and_save_plot(
        plot_data=analysis_types["Unadjusted"]["data_list"],
        plot_title="Unadjusted",
        output_filename_base="unadjusted_forest_plot"
    )
    
    generate_and_save_plot(
        plot_data=analysis_types["Adjusted"]["data_list"],
        plot_title="Adjusted",
        output_filename_base="adjusted_forest_plot"
    )

    console.print("\n[bold magenta]--- Script finished. ---[/bold magenta]")

if __name__ == "__main__":
    main()