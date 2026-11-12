<#
.SYNOPSIS
    Apply and verify the Colab CLI agent-quota fixes on this machine.

.DESCRIPTION
    One re-runnable command that fixes the three problems that made OMP
    agents burn quota while driving the Colab CLI, and proves the fixes
    work:

      1. The two-interpreter trap. The system Python (C:\Python313) had
         *upstream* google-colab-cli 0.7.4 installed, whose unguarded
         `import termios` crashes every `colab` command on Windows, and a
         hand-written fake `termios.py` had been dropped into the stdlib to
         work around it. The fork is installed (editable) into the system
         Python instead, the fake shim is removed, and the stale uv launcher
         on PATH (which exits 103 with "untrusted mount point") is retired.

      2. `--timeout` did not bound wall-clock. `jupyter_kernel_client` turns
         the deadline into a 100% CPU spin that never raises, so a hung cell
         kept the Colab VM billing forever. The fork now enforces the
         deadline itself and interrupts the kernel.

      3. `colab new` asked for a second GPU runtime even when one was
         already running, and the account's concurrency limit refused it
         (HTTP 412). The fork now adopts the live runtime and rotates to
         another signed-in account when the quota is exhausted.

    Every run appends to a change log so the history of what was changed on
    this machine is auditable.

.PARAMETER SkipTests
    Skip the pytest run (the verification of the code changes).

.PARAMETER SkipInstall
    Skip the pip install into the system Python (code changes only).

.PARAMETER DryRun
    Report what would change without touching anything.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\apply-agent-fixes.ps1
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$SkipInstall,
    [switch]$DryRun
)

# NOTE ON $ErrorActionPreference
#   Windows PowerShell 5.1 (the `powershell` executable) turns a native
#   command's stderr into a *terminating* error when the preference is
#   'Stop' -- even when stderr is redirected to a file. The Colab CLI writes
#   notices to stderr on purpose (stdout stays machine-parseable), so that
#   would abort every check with an empty message. PowerShell 7 does not do
#   this. The script therefore runs with 'Continue' and signals real errors
#   explicitly with try/catch and exit-code checks.
$ErrorActionPreference = 'Continue'

$RepoRoot    = Split-Path -Parent $PSScriptRoot
$ForkVenvPy  = 'D:\colab-win-venv\Scripts\python.exe'
$SystemPy    = 'C:\Python313\python.exe'
$SystemDir   = 'C:\Python313'
$ColabExe    = Join-Path $SystemDir 'Scripts\colab.exe'
$FakeTermios = Join-Path $SystemDir 'Lib\termios.py'
$StaleShim   = Join-Path $env:USERPROFILE '.local\bin\colab.exe'
$BackupDir   = Join-Path $env:USERPROFILE '.config\colab-cli\backup'
$ChangeLog   = Join-Path $env:USERPROFILE '.config\colab-cli\CHANGELOG.agent-fixes.md'

$script:Results = @()

function Write-Step { param([string]$m) Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host "    OK   $m" -ForegroundColor Green }
function Write-Warn { param([string]$m) Write-Host "    WARN $m" -ForegroundColor Yellow }
function Write-Fail { param([string]$m) Write-Host "    FAIL $m" -ForegroundColor Red }

function Add-Result {
    param([string]$Step, [string]$Status, [string]$Detail)
    $script:Results += [pscustomobject]@{ Step = $Step; Status = $Status; Detail = $Detail }
}

# Run a native command, capturing stdout and stderr separately.
#
# The output is captured with Out-String rather than piped into
# Select-Object -First: that closes the pipe early and kills the command with
# a broken-pipe exit code.
function Invoke-Native {
    param(
        [string]$Exe,
        [string[]]$NativeArgs = @(),
        [int[]]$OkExit = @(0)
    )
    $errFile = [IO.Path]::GetTempFileName()
    try {
        $out = & $Exe @NativeArgs 2>$errFile | Out-String
        $code = $LASTEXITCODE
        $err = ''
        if (Test-Path $errFile) {
            $err = (Get-Content $errFile -Raw -ErrorAction SilentlyContinue)
        }
    } finally {
        Remove-Item $errFile -ErrorAction SilentlyContinue
    }
    if ($OkExit -notcontains $code) {
        throw "exit $code :: $out $err"
    }
    return $out
}

# Run the CLI and throw unless the exit code is acceptable.
function Invoke-Cli {
    param([string[]]$CliArgs, [int[]]$OkExit = @(0))
    return Invoke-Native -Exe $ColabExe -NativeArgs $CliArgs -OkExit $OkExit
}

# Run a check body. The body signals failure by throwing.
function Invoke-Check {
    param([string]$Name, [scriptblock]$Body)
    try {
        $null = & $Body
        Write-Ok $Name
        Add-Result $Name 'ok' ''
        return $true
    } catch {
        Write-Fail "$Name :: $($_.Exception.Message)"
        Add-Result $Name 'fail' $_.Exception.Message
        return $false
    }
}

# ---------------------------------------------------------------- preflight

Write-Step "Preflight"
if (-not (Test-Path $RepoRoot))  { throw "Repo root not found: $RepoRoot" }
if (-not (Test-Path $ForkVenvPy)) { throw "Fork venv python not found: $ForkVenvPy" }
if (-not (Test-Path $SystemPy))   { throw "System python not found: $SystemPy" }
New-Item -ItemType Directory -Force -Path $BackupDir | Out-Null
Write-Ok "repo=$RepoRoot"
Write-Ok "fork venv=$ForkVenvPy"
Write-Ok "system python=$SystemPy"

if ($DryRun) {
    Write-Host ""
    Write-Host "DRY RUN - no changes will be made." -ForegroundColor Yellow
    Write-Host "Would:"
    if (Test-Path $FakeTermios) { Write-Host "  - back up and remove $FakeTermios" }
    if (Test-Path $StaleShim)   { Write-Host "  - back up and remove $StaleShim (uv launcher, exits 103)" }
    Write-Host "  - uninstall upstream google-colab-cli from $SystemDir"
    Write-Host "  - pip install -e $RepoRoot into $SystemDir"
    Write-Host "  - run the test suite with $ForkVenvPy"
    exit 0
}

# ------------------------------------------------- 1. remove the fake shim

Write-Step "Removing the fake stdlib termios shim"
if (Test-Path $FakeTermios) {
    $dest = Join-Path $BackupDir ("termios.py." + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Copy-Item $FakeTermios $dest -Force
    Remove-Item $FakeTermios -Force
    Write-Ok "removed $FakeTermios (backup: $dest)"
    Add-Result 'remove fake stdlib termios.py' 'ok' "backup: $dest"
} else {
    Write-Ok "not present (already clean)"
    Add-Result 'remove fake stdlib termios.py' 'ok' 'already absent'
}

# ------------------------------------- 2. retire the broken uv launcher

Write-Step "Retiring the broken uv 'colab' launcher on PATH"
if (Test-Path $StaleShim) {
    $dest = Join-Path $BackupDir ("colab.exe.uvshim." + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    Copy-Item $StaleShim $dest -Force
    Remove-Item $StaleShim -Force
    Write-Ok "removed $StaleShim (backup: $dest)"
    Add-Result 'remove broken uv colab launcher' 'ok' "backup: $dest"
} else {
    Write-Ok "not present (already clean)"
    Add-Result 'remove broken uv colab launcher' 'ok' 'already absent'
}

# --------------------------------- 3. install the fork into system python

if (-not $SkipInstall) {
    Write-Step "Installing the fork into the system Python"
    & $SystemPy -m pip uninstall -y google-colab-cli 2>&1 | ForEach-Object { Write-Host "    $_" }
    & $SystemPy -m pip install --disable-pip-version-check -e $RepoRoot 2>&1 |
        Select-Object -Last 4 | ForEach-Object { Write-Host "    $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "pip install failed (exit $LASTEXITCODE)"
        Add-Result 'install fork into system python' 'fail' "exit $LASTEXITCODE"
    } else {
        Write-Ok "installed editable from $RepoRoot"
        Add-Result 'install fork into system python' 'ok' $RepoRoot
    }
} else {
    Write-Step "Skipping the install (-SkipInstall)"
}

# ------------------------------- 4. install the fork into every environment

Write-Step "Installing the fork into every environment that has a colab_cli"

# Find every site-packages directory that carries a colab_cli package, then
# install this checkout (editable) into the environment it belongs to, so
# every Python on the box runs the SAME custom build.
$envs = @()
foreach ($root in @('C:\', 'D:\')) {
    Get-ChildItem $root -Directory -Recurse -Depth 6 -Force -Filter 'colab_cli' `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match 'site-packages\\colab_cli$' } |
        ForEach-Object {
            $sitePackages = Split-Path $_.FullName -Parent
            $envRoot = Split-Path (Split-Path $sitePackages -Parent) -Parent
            $py = Join-Path $envRoot 'Scripts\python.exe'
            if (Test-Path $py) {
                $envs += [pscustomobject]@{ Env = $envRoot; Python = $py }
            }
        }
}
$envs = $envs | Sort-Object Env -Unique

foreach ($e in $envs) {
    # The package needs Python >= 3.12. Older environments cannot host it;
    # that is a fact about them, not a failure of this script.
    $ver = & $e.Python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>&1 | Out-String
    $ver = $ver.Trim()
    if ($ver -match '^(\d+)\.(\d+)' -and ([int]$matches[1] -lt 3 -or ($matches[1] -eq 3 -and [int]$matches[2] -lt 12))) {
        Write-Warn "$($e.Env) runs Python $ver (needs >= 3.12) - skipped"
        Add-Result "install into $($e.Env)" 'warn' "python $ver < 3.12"
        continue
    }

    # uv-made venvs often have no pip; uv can install into them directly.
    $hasPip = $true
    & $e.Python -m pip --version *> $null
    if ($LASTEXITCODE -ne 0) { $hasPip = $false }

    if ($hasPip) {
        $out = & $e.Python -m pip install --disable-pip-version-check -q -e $RepoRoot 2>&1
    } else {
        $out = & uv pip install --python $e.Python -e $RepoRoot 2>&1
    }
    if ($LASTEXITCODE -ne 0) {
        ($out | Select-Object -Last 3) | ForEach-Object { Write-Host "    $_" }
        Write-Fail "install failed for $($e.Env)"
        Add-Result "install into $($e.Env)" 'fail' "exit $LASTEXITCODE"
    } else {
        Write-Ok "$($e.Env) -> this checkout"
        Add-Result "install into $($e.Env)" 'ok' $RepoRoot
    }
}
if ($envs.Count -eq 0) {
    Write-Warn "no other environments with colab_cli were found"
}

# ------------------------------------------------------------ verification

Write-Step "Verifying"

Invoke-Check 'system python imports the fork' {
    $out = Invoke-Native -Exe $SystemPy -NativeArgs @('-c', "import colab_cli.commands.session as s, colab_cli.commands.account as a, colab_cli.accounts as ac; print(s.__file__); print(a.__file__); print(ac.__file__)")
    # The editable install puts the *code* under the repo; site-packages only
    # holds the package data (README/SKILL), so check the module files.
    if ($out -notmatch 'google-colab-cli-win') { throw "not importing the fork: $out" }
} | Out-Null

Invoke-Check 'no fake termios.py in the stdlib' {
    if (Test-Path $FakeTermios) { throw "still present: $FakeTermios" }
} | Out-Null

Invoke-Check 'colab on PATH resolves to the system scripts dir' {
    $resolved = (Get-Command colab -ErrorAction Stop).Source
    if ($resolved -notlike "$SystemDir*") {
        throw "colab resolves to $resolved, expected something under $SystemDir"
    }
} | Out-Null

Invoke-Check 'colab-tool is the command on PATH' {
    $resolved = (Get-Command colab-tool -ErrorAction Stop).Source
    if ($resolved -notlike "$SystemDir*") {
        throw "colab-tool resolves to $resolved, expected something under $SystemDir"
    }
} | Out-Null

Invoke-Check 'colab --help works' {
    $null = Invoke-Cli @('--help')
} | Out-Null

Invoke-Check 'colab help says this is a custom build' {
    $out = Invoke-Cli @('help')
    if ($out -notmatch 'CUSTOM Colab CLI') { throw "the help does not identify the build" }
    if ($out -notmatch 'agent-guide') { throw "the help does not point at agent-guide" }
} | Out-Null

Invoke-Check 'colab account list works' {
    $null = Invoke-Cli @('account', 'list')
} | Out-Null

Invoke-Check 'colab account list --json is machine readable' {
    $out = Invoke-Cli @('account', 'list', '--json')
    $null = $out | ConvertFrom-Json
} | Out-Null

Invoke-Check 'colab --account accepts a label' {
    $null = Invoke-Cli @('--account', 'work', 'account', 'list')
} | Out-Null

Invoke-Check 'colab --account rejects a bad label' {
    $null = Invoke-Cli @('--account', 'bad label', 'account', 'list') -OkExit @(2)
} | Out-Null

Invoke-Check 'colab doctor reports a verdict' {
    $out = Invoke-Cli @('doctor', '--json')
    $doc = $out | ConvertFrom-Json
    if ($null -eq $doc.ok) { throw "doctor did not report a verdict: $out" }
} | Out-Null

Invoke-Check 'the timeout guard is in the installed fork' {
    $null = Invoke-Native -Exe $SystemPy -NativeArgs @('-c', "import inspect, colab_cli.runtime as r; src=inspect.getsource(r); assert '_run_with_deadline' in src and 'ExecutionTimeout' in src; print('ok')")
} | Out-Null

Invoke-Check 'the account rotation is in the installed fork' {
    $null = Invoke-Native -Exe $SystemPy -NativeArgs @('-c', "import inspect, colab_cli.commands.session as s; src=inspect.getsource(s); assert '_assign_with_failover' in src and '_adopt_existing_session' in src; print('ok')")
} | Out-Null

if (-not $SkipTests) {
    Invoke-Check 'test suite is green' {
        Push-Location $RepoRoot
        try {
            $null = Invoke-Native -Exe $ForkVenvPy -NativeArgs @('-m', 'pytest', 'tests/', '-q', '-p', 'no:cacheprovider')
        } finally {
            Pop-Location
        }
    } | Out-Null
}

# --------------------------------------------------------------- changelog

Write-Step "Writing the change log"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ChangeLog) | Out-Null
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'
$log = @()
$log += "## Run $stamp"
$log += ""
$log += "Repo: $RepoRoot"
$log += ""
$log += "| Step | Status | Detail |"
$log += "| --- | --- | --- |"
foreach ($r in $script:Results) {
    $detail = (($r.Detail -replace '\|', '\|') -replace "`r?`n", ' ').Trim()
    $log += "| $($r.Step) | $($r.Status) | $detail |"
}
$failed = @($script:Results | Where-Object { $_.Status -eq 'fail' })
$log += ""
if ($failed.Count -eq 0) {
    $log += "**Result: all checks passed.**"
} else {
    $log += "**Result: $($failed.Count) check(s) FAILED.**"
}
$log += ""
Add-Content -Path $ChangeLog -Value ($log -join "`n") -Encoding UTF8
Write-Ok "appended to $ChangeLog"

Write-Host ""
if ($failed.Count -eq 0) {
    Write-Host "ALL CHECKS PASSED" -ForegroundColor Green
    exit 0
}
Write-Host "$($failed.Count) CHECK(S) FAILED - see $ChangeLog" -ForegroundColor Red
exit 1
