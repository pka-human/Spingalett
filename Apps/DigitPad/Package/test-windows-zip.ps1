# SPDX-License-Identifier: MIT
#
# Tests a DigitPad Windows zip on Windows: unpacks it into a folder with a non-ASCII name,
# classifies seven.pgm from the command line, then opens the window, draws a 7 with the mouse and
# reads the prediction DigitPad prints. Only Windows itself is on the DLL search path, so every
# other DLL must come from the zip.
#
#   pwsh Apps/DigitPad/Package/test-windows-zip.ps1 DigitPad-<version>-windows-x86_64.zip [-Screenshot out.png]

param(
    [Parameter(Mandatory)] [string] $Zip,
    [string] $Screenshot
)
$ErrorActionPreference = 'Stop'

$test = Join-Path ([IO.Path]::GetTempPath()) ('digitpad-' + -join [char[]](0x442, 0x435, 0x441, 0x442))
Remove-Item $test -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $Zip $test
$exe = (Get-ChildItem $test -Recurse -Filter DigitPad.exe).FullName
$sample = Join-Path $test 'seven.pgm'
Copy-Item (Join-Path $PSScriptRoot 'seven.pgm') $sample
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
Write-Host "testing $exe"

function Read-Text([string] $path) { if (Test-Path $path) { (Get-Content $path -Raw) ?? '' } else { '' } }

# ---- 1. command line: started from another directory, it finds the model next to itself
$out = Join-Path $test 'classify.txt'
$err = Join-Path $test 'classify-err.txt'
$p = Start-Process $exe -ArgumentList '--classify', "`"$sample`"" -WorkingDirectory $env:SystemRoot `
    -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait -PassThru
$text = Read-Text $out
Write-Host "--classify: exit $($p.ExitCode): $text$(Read-Text $err)"
if ($p.ExitCode -ne 0 -or $text -notmatch '^prediction 7 ') { throw '--classify did not recognise the 7' }

# ---- 2. the window
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class Desktop {
    [StructLayout(LayoutKind.Sequential)] public struct Point { public int X, Y; }
    [StructLayout(LayoutKind.Sequential)] public struct Rect { public int Left, Top, Right, Bottom; }
    [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr window, ref Point point);
    [DllImport("user32.dll")] public static extern bool GetClientRect(IntPtr window, out Rect rect);
    [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr window);
    [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
    [DllImport("user32.dll")] public static extern void mouse_event(uint flags, int dx, int dy, uint data, UIntPtr extra);
    [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr window, uint message, IntPtr w, IntPtr l);
}
'@

$out = Join-Path $test 'gui.txt'
$err = Join-Path $test 'gui-err.txt'
$p = Start-Process $exe -ArgumentList '--verbose' -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$null = $p.Handle                   # keeps the handle, and with it the exit code, once it exits
$deadline = (Get-Date).AddSeconds(60)
do {
    Start-Sleep -Milliseconds 250
    $p.Refresh()
} until ($p.HasExited -or $p.MainWindowTitle -eq 'DigitPad - Spingalett' -or (Get-Date) -gt $deadline)
if ($p.MainWindowTitle -ne 'DigitPad - Spingalett') {
    throw "no DigitPad window (exited: $($p.HasExited), title: '$($p.MainWindowTitle)'): $(Read-Text $err)"
}
$window = $p.MainWindowHandle
Start-Sleep -Seconds 1
[void][Desktop]::SetForegroundWindow($window)

# the 880x584 interface fills the client area (DigitPad doubles it on tall screens)
$origin = New-Object Desktop+Point
[void][Desktop]::ClientToScreen($window, [ref]$origin)
$client = New-Object Desktop+Rect
[void][Desktop]::GetClientRect($window, [ref]$client)
$scale = $client.Right / 880.0
Write-Host "window at $($origin.X),$($origin.Y), client $($client.Right)x$($client.Bottom)"
function Move-To([double] $x, [double] $y) {     # a point on the 400x400 canvas at (24, 84)
    [void][Desktop]::SetCursorPos([int]($origin.X + (24 + $x) * $scale), [int]($origin.Y + (84 + $y) * $scale))
    Start-Sleep -Milliseconds 8
}

$stroke = @(@(128, 96), @(272, 96), @(188, 322))
Move-To $stroke[0][0] $stroke[0][1]
[Desktop]::mouse_event(0x0002, 0, 0, 0, [UIntPtr]::Zero)          # left button down
for ($i = 1; $i -lt $stroke.Count; $i++) {
    $a = $stroke[$i - 1]; $b = $stroke[$i]
    for ($t = 1; $t -le 30; $t++) { Move-To ($a[0] + ($b[0] - $a[0]) * $t / 30) ($a[1] + ($b[1] - $a[1]) * $t / 30) }
}
[Desktop]::mouse_event(0x0004, 0, 0, 0, [UIntPtr]::Zero)          # left button up
Start-Sleep -Seconds 1

if ($Screenshot) {
    try {                           # for a look at the result; the test does not depend on it
        Add-Type -AssemblyName System.Drawing
        $image = [System.Drawing.Bitmap]::new($client.Right, $client.Bottom)
        $graphics = [System.Drawing.Graphics]::FromImage($image)
        $graphics.CopyFromScreen($origin.X, $origin.Y, 0, 0, $image.Size)
        $image.Save((Join-Path (Get-Location) $Screenshot))
        $graphics.Dispose(); $image.Dispose()
    } catch { Write-Warning "no screenshot: $_" }
}
[void][Desktop]::PostMessage($window, 0x0010, [IntPtr]::Zero, [IntPtr]::Zero)   # WM_CLOSE
if (-not $p.WaitForExit(15000)) { $p.Kill(); throw 'DigitPad did not close' }

$text = Read-Text $out
Write-Host "window: exit $($p.ExitCode): $text$(Read-Text $err)"
if ($p.ExitCode -ne 0) { throw "DigitPad exited with $($p.ExitCode)" }
if ($text -notmatch '(?m)^prediction 7 ') { throw 'the 7 drawn in the window was not recognised' }
Write-Host 'DigitPad works'
