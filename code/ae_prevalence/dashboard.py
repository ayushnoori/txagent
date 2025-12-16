import streamlit as st
import pandas as pd
from pathlib import Path
import re

# --- Dashboard Configuration ---

# This path should point to the root directory where your analysis results are saved.
# It's the parent folder of the 'ae_prevalence' directory from your original script.
# Example: //10.100.117.220/Research_Archive$/Archive/R01/R01-Ayush/results
# Using a raw string (r"...") is recommended for Windows paths.
RESULTS_BASE_DIR = Path(r"//10.100.117.220/Research_Archive$/Archive/R01/R01-Ayush/results")
ANALYSIS_RESULTS_DIR = RESULTS_BASE_DIR / "ae_prevalence"


# --- Helper Functions ---

def format_selection_name(path: Path) -> str:
    """Converts a path's folder names into a readable string for display."""
    # Takes the last two parts of the path (e.g., 'hypertension' and 'hypertension-chronic_kidney_disease-arb')
    parts = path.parts[-2:]
    
    # Cleans up each part by replacing underscores/hyphens and capitalizing
    cleaned_parts = [re.sub(r'[_ -]', ' ', part).title() for part in parts]
    
    return "  Kombination: ".join(cleaned_parts)


@st.cache_data
def find_analysis_runs(results_dir: Path) -> dict:
    """
    Scans the results directory to find all completed analysis runs.
    
    An analysis run is considered valid if it contains at least one of the expected
    result files (prevalence, odds ratio, or a barplot).
    
    Returns:
        A dictionary mapping a display-friendly name to its corresponding folder path.
    """
    analysis_paths = {}
    if not results_dir.is_dir():
        return analysis_paths

    # The structure is /<disease>/<analysis_slug>, so we need to glob two levels deep.
    for path in results_dir.glob("*/*"):
        if path.is_dir():
            # Check if any key result files or folders exist
            has_or_csv = (path / "odds_ratio_results.csv").exists()
            has_prev_csv = (path / "prevalence_results.csv").exists()
            has_plots_dir = (path / "barplots").is_dir()
            
            if has_or_csv or has_prev_csv or has_plots_dir:
                display_name = format_selection_name(path)
                analysis_paths[display_name] = path
                
    return analysis_paths


# --- Main Dashboard Application ---

def main():
    """Renders the Streamlit dashboard."""
    
    # Configure the page layout to be wide
    st.set_page_config(
        page_title="Adverse Event Analysis Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🔬 Adverse Event Analysis Dashboard")
    st.markdown("Use the dropdown menu in the sidebar to select and view an analysis run.")

    # --- Sidebar for Selection ---
    with st.sidebar:
        st.header("Analysis Selection")
        analysis_runs = find_analysis_runs(ANALYSIS_RESULTS_DIR)

        if not analysis_runs:
            st.error(
                f"No analysis results found in the specified directory:\n\n"
                f"{ANALYSIS_RESULTS_DIR}\n\n"
                "Please verify the `RESULTS_BASE_DIR` path in the script and ensure the analysis has been run."
            )
            return

        # Create a dropdown menu with the user-friendly names
        sorted_runs = sorted(analysis_runs.keys())
        selected_run_name = st.selectbox(
            "Choose an analysis to display:",
            options=sorted_runs,
            index=0
        )

    # --- Main Panel for Displaying Results ---
    
    # Get the path corresponding to the selected run name
    selected_run_path = analysis_runs[selected_run_name]

    st.header(f"Showing Results For: `{selected_run_name}`")
    st.write(f"**Source Directory:** `{selected_run_path}`")

    # Define paths for the expected result files
    or_results_file = selected_run_path / "odds_ratio_results.csv"
    prevalence_results_file = selected_run_path / "prevalence_results.csv"
    plots_dir = selected_run_path / "barplots"
    
    st.divider()

    # Create two columns for the data tables
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Prevalence Results")
        if prevalence_results_file.exists():
            try:
                prevalence_df = pd.read_csv(prevalence_results_file)
                st.dataframe(prevalence_df)
            except Exception as e:
                st.error(f"Failed to load prevalence results: {e}")
        else:
            st.warning("Prevalence results file not found for this analysis.")

    with col2:
        st.subheader("📈 Odds Ratio (OR) Results")
        if or_results_file.exists():
            try:
                or_df = pd.read_csv(or_results_file)
                st.dataframe(or_df)
            except Exception as e:
                st.error(f"Failed to load odds ratio results: {e}")
        else:
            st.warning("Odds ratio results file not found for this analysis.")

    st.divider()
    
    # --- Bar Plots Section ---
    st.subheader("🖼️ Prevalence Bar Plots")
    if plots_dir.is_dir():
        # Find all PNG files in the barplots directory
        plot_files = sorted(list(plots_dir.glob("*.png")))
        
        if not plot_files:
            st.info("The 'barplots' directory exists but contains no plot images.")
        else:
            # Create a responsive grid of columns to display plots
            num_columns = 3
            cols = st.columns(num_columns)
            for i, plot_file in enumerate(plot_files):
                with cols[i % num_columns]:
                    st.image(
                        str(plot_file),
                        caption=plot_file.stem.replace('-', ' ').replace('_', ' ').title(),
                        use_column_width=True
                    )
    else:
        st.warning("Bar plots directory not found for this analysis.")


if __name__ == "__main__":
    main()
