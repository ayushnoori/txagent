# =============================================================================
#  PowerShell Setup and Execution Script
# =============================================================================

# --- Step 1: Map Network Drive (if not already mapped) ---
Write-Host "Checking for network drive Z:..." -ForegroundColor Yellow
$drivePath = "\\10.10.117.220\Research_Archive$\Archive\R01\R01-Ayush"

if (-not (Test-Path -Path "Z:")) {
    Write-Host "Drive Z: not found. Mapping network drive..."
    try {
        net use Z: $drivePath
        Write-Host "Successfully mapped Z: to $drivePath" -ForegroundColor Green
    } catch {
        Write-Host "Error: Failed to map network drive. Please check connection and permissions." -ForegroundColor Red
        # Pause to allow user to read the error, then exit
        Read-Host "Press Enter to exit..."
        exit 1
    }
} else {
    Write-Host "Drive Z: is already mapped." -ForegroundColor Green
}

# --- Step 2: Configure Environment ---
Write-Host "`nSetting up Python environment..." -ForegroundColor Yellow

# Configure pip and install uv (these commands will only run if needed)
pip config --user set global.index-url https://v-ayushno_nat:cmVmdGtu(...) # Truncated for display
pip config --user set global.trusted-host jfrog.apps.ocpdmzp.wclalit.org.il
pip install uv

# Synchronize the environment using the pyproject.toml file
uv sync
Write-Host "Environment setup complete." -ForegroundColor Green


# --- Step 3: Find Python Scripts and Build Menu ---
Write-Host "`nSearching for Python scripts in the 'code' directory..." -ForegroundColor Yellow
$pythonScripts = Get-ChildItem -Path ".\code" -Filter "*.py" -Recurse

# --- Step 4: Display Menu and Get User Input ---
while ($true) {
    Write-Host "`n================ MENU ================" -ForegroundColor Cyan
    Write-Host " 0: Run Jupyter Lab"
    
    # Loop through found scripts to create menu items
    for ($i = 0; $i -lt $pythonScripts.Count; $i++) {
        # Display the script name and its parent directory for context
        $scriptName = $pythonScripts[$i].Name
        $scriptParentDir = $pythonScripts[$i].Directory.Name
        Write-Host " $($i + 1): Run '$scriptName' (in '$scriptParentDir')"
    }
    
    $noneOption = $pythonScripts.Count + 1
    Write-Host " ${noneOption}: None (Exit)"
    Write-Host "====================================" -ForegroundColor Cyan

    $choice = Read-Host "Enter your selection"

    # --- Step 5: Execute Command Based on Selection ---
    if ($choice -eq "0") {
        Write-Host "`nLaunching Jupyter Lab..." -ForegroundColor Green
        uv run --with jupyter jupyter lab
        break
    }
    elseif ($choice -eq $noneOption) {
        Write-Host "Exiting."
        break
    }
    # Check if the choice is a number corresponding to a script
    elseif ($choice -match '^\d+$' -and [int]$choice -ge 1 -and [int]$choice -le $pythonScripts.Count) {
        $scriptIndex = [int]$choice - 1
        $selectedScript = $pythonScripts[$scriptIndex]
        
        # Change to the script's directory before running
        $targetDir = $selectedScript.DirectoryName
        $scriptFile = $selectedScript.Name
        
        Write-Host "`nChanging directory to '$targetDir'" -ForegroundColor Yellow
        Set-Location -Path $targetDir
        
        Write-Host "Executing 'uv run python $scriptFile'..." -ForegroundColor Green
        uv run python $scriptFile
        
        # Optional: Change back to the original directory after execution
        # Set-Location -Path $PSScriptRoot
        break
    }
    else {
        Write-Host "`nInvalid selection. Please try again." -ForegroundColor Red
    }
}
