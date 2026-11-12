<#
.SYNOPSIS
    Run Colab CLI jobs that need a real TTY, or that must outlive one tool call.

.DESCRIPTION
    Two things the Colab CLI needs that a plain piped shell cannot give it:

      1. A real TTY. `colab account add <label>` runs the copy-paste OAuth
         flow and calls input() for the authorization code. Without a TTY
         stdin hits EOF and the sign-in dies right after printing the URL.

      2. A process that outlives the tool call. A long `colab exec` /
         `colab run` in a plain shell blocks the caller and is killed with
         it; in a psmux session it keeps running and its output can be
         collected later.

    Modes:
      auth  <label>        sign in an additional Google account (needs a human)
      job   <name> <args>  run a colab command in a persistent psmux session
      log   <name>          show the captured output of a job
      list                  list the psmux sessions this script created
      kill  <name>         tear down a session

.EXAMPLE
    # sign in a second account
    .\scripts\colab-psmux.ps1 auth work
    # (visit the URL, then)
    .\scripts\colab-psmux.ps1 auth work -Code 4/0Abc...

    # run a long job in the background
    .\scripts\colab-psmux.ps1 job train -- exec -s train -f job.py --json
    .\scripts\colab-psmux.ps1 log train
#>
[CmdletBinding(DefaultParameterSetName = 'job')]
param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$Mode,

    [Parameter(Position = 1)]
    [string]$Name = 'colab',

    [Parameter(Position = 2, ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @(),

    [string]$Code,
    [switch]$Wait
)

$ErrorActionPreference = 'Continue'

# PowerShell tries to bind a bare `--` as a parameter name, so drop a leading
# separator that users copy from tmux habits.
if ($Rest -and $Rest.Count -gt 0 -and $Rest[0] -eq '--') {
    $Rest = $Rest[1..($Rest.Count - 1)]
}

$ColabExe = 'C:\Python313\Scripts\colab.exe'
$LogDir   = Join-Path $env:USERPROFILE '.config\colab-cli\psmux-logs'
$Prefix   = 'colab-'

function Write-Step { param([string]$m) Write-Host "==> $m" -ForegroundColor Cyan }
function Write-Ok   { param([string]$m) Write-Host "    $m" -ForegroundColor Green }
function Write-Err  { param([string]$m) Write-Host "    $m" -ForegroundColor Red }

# psmux's own output must not be swallowed by the caller's pipeline.
function Invoke-Psmux {
    param([string[]]$PsmuxArgs)
    $out = & psmux @PsmuxArgs 2>&1 | Out-String
    return $out
}

function Get-Session {
    param([string]$Session)
    & psmux has-session -t $Session *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-Pane {
    param([string]$Session)
    $raw = Invoke-Psmux @('capture-pane', '-t', $Session, '-p')
    # Strip control characters and drop blank lines so the output is readable.
    ($raw -split "`r?`n" |
        ForEach-Object { ($_ -replace '[^\x20-\x7E]', '') } |
        Where-Object { $_.Trim() -ne '' })
}

switch ($Mode.ToLower()) {

    'auth' {
        $label = if ($Name) { $Name } else { 'account' }
        $session = "$Prefix$label-auth"
        $log = Join-Path $LogDir "$session.log"

        Write-Step "Starting the OAuth sign-in for account '$label' in psmux session '$session'"
        if (Get-Session $session) {
            Write-Ok "session '$session' already exists - reusing it"
        } else {
            New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
            Invoke-Psmux @(
                'new-session', '-d', '-s', $session, '-x', '200', '-y', '50',
                '-c', $env:USERPROFILE, '--', $ColabExe, 'account', 'add', $label
            ) | Out-Null
            Start-Sleep -Seconds 2
        }

        # Wait for the authorization URL to appear.
        $url = $null
        for ($i = 0; $i -lt 30 -and -not $url; $i++) {
            $pane = Get-Pane $session
            $url = ($pane | Where-Object { $_ -match 'https://accounts\.google\.com/o/oauth2/auth\?' } | Select-Object -First 1)
            if (-not $url) {
                # Show the tail so a failure is visible.
                $fatal = $pane | Where-Object { $_ -match 'Traceback|Error:' } | Select-Object -First 1
                if ($fatal) {
                    Write-Err $fatal
                    break
                }
                Start-Sleep -Seconds 1
            }
        }

        if (-not $url) {
            Write-Err "no authorization URL appeared in '$session'"
            Get-Pane $session | Select-Object -Last 15 | ForEach-Object { Write-Host "    $_" }
            exit 1
        }

        Write-Host ""
        Write-Host "    Open this URL in a browser and approve the sign-in:" -ForegroundColor Yellow
        Write-Host "    $url" -ForegroundColor White
        Write-Host ""

        if ($Code) {
            Write-Step "Sending the authorization code"
            Invoke-Psmux @('send-keys', '-t', $session, $Code, 'Enter') | Out-Null
            Start-Sleep -Seconds 4
            Write-Ok "code sent to '$session'"
            Get-Pane $session | Select-Object -Last 8 | ForEach-Object { Write-Host "    $_" }
        } else {
            Write-Host "    Then either:" -ForegroundColor Yellow
            Write-Host "      a) run this again with -Code <the code>:" -ForegroundColor Yellow
            Write-Host "         .\scripts\colab-psmux.ps1 auth $label -Code 4/0Abc..." -ForegroundColor White
            Write-Host "      b) or attach and paste it yourself:" -ForegroundColor Yellow
            Write-Host "         psmux attach -t $session" -ForegroundColor White
            Write-Host ""
            Write-Host "    The session stays alive until it is killed." -ForegroundColor Yellow
        }
    }

    'job' {
        $session = if ($Name.StartsWith($Prefix)) { $Name } else { "$Prefix$Name" }
        $log = Join-Path $LogDir "$session.log"
        $wrapper = Join-Path $LogDir "$session.cmd"

        if (Get-Session $session) {
            Write-Err "session '$session' already exists (kill it first: ... kill $session)"
            exit 1
        }

        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
        if (Test-Path $log) { Remove-Item $log -Force }

        $args = @()
        if ($Rest) { $args = $Rest }
        $argline = ($args | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join ' '

        # A generated .cmd wrapper avoids all the quoting problems of passing
        # an argument list through psmux, and gives the job a log file that
        # survives the pane.
        @(
            '@echo off',
            "`"$ColabExe`" $argline > `"$log`" 2>&1",
            "echo EXITCODE=%ERRORLEVEL% >> `"$log`""
        ) | Set-Content -Path $wrapper -Encoding ASCII

        Write-Step "Starting '$session': colab $($args -join ' ')"
        Invoke-Psmux @(
            'new-session', '-d', '-s', $session, '-x', '200', '-y', '50',
            '-c', $env:USERPROFILE, '--', 'cmd', '/c', $wrapper
        ) | Out-Null
        Start-Sleep -Seconds 1

        Write-Ok "session '$session' is running"
        Write-Host "    log: $log"
        Write-Host "    tail it:   .\scripts\colab-psmux.ps1 log $session"
        Write-Host "    tear down: .\scripts\colab-psmux.ps1 kill $session"

        if ($Wait) {
            while (Get-Session $session) { Start-Sleep -Seconds 2 }
            Write-Step "job finished"
            Get-Content $log -Tail 40 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" }
        }
    }

    'log' {
        $session = if ($Name.StartsWith($Prefix)) { $Name } else { "$Prefix$Name" }
        $log = Join-Path $LogDir "$session.log"
        Write-Step "Log for '$session'"
        if (Test-Path $log) {
            Get-Content $log -Tail 60 | ForEach-Object { Write-Host "    $_" }
        } else {
            Write-Host "    (no log yet at $log)"
            if (Get-Session $session) {
                Write-Host "    live pane:"
                Get-Pane $session | Select-Object -Last 30 | ForEach-Object { Write-Host "    $_" }
            }
        }
    }

    'list' {
        Write-Step "psmux sessions"
        $ls = Invoke-Psmux @('ls')
        if ($ls.Trim()) {
            $ls -split "`r?`n" | Where-Object { $_.Trim() } | ForEach-Object { Write-Host "    $_" }
        } else {
            Write-Host "    (none)"
        }
        Write-Step "logs in $LogDir"
        Get-ChildItem $LogDir -Filter *.log -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 20 |
            ForEach-Object { Write-Host ("    {0}  {1}" -f $_.LastWriteTime.ToString('HH:mm:ss'), $_.Name) }
    }

    'kill' {
        $session = if ($Name.StartsWith($Prefix)) { $Name } else { "$Prefix$Name" }
        if (Get-Session $session) {
            Invoke-Psmux @('kill-session', '-t', $session) | Out-Null
            Write-Ok "killed '$session'"
        } else {
            Write-Ok "'$session' was not running"
        }
    }

    default {
        Write-Err "unknown mode '$Mode' (use auth|job|log|list|kill)"
        exit 2
    }
}
