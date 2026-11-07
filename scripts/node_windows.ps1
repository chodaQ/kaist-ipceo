# 와트투비트 노드 원클릭 실행 (Windows) — start_laptop_windows.bat 이 호출
#   처음 실행: 오케스트레이터 주소·토큰·노드 ID 를 묻고 .w2b\node.json 에 저장
#   필요하면 llama.cpp(CUDA 빌드)·모델을 .w2b\ 에 내려받음 (끊겨도 이어받기)
#   점검(agent.py --check) 후 에이전트 실행, 죽으면 5초 뒤 자동 재시작 (창 닫기 / Ctrl+C 로 종료)
param([switch]$Reconfigure, [switch]$CheckOnly, [switch]$SkipCheck)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch {}
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent $PSScriptRoot
$W = Join-Path $Root ".w2b"
$Conf = Join-Path $W "node.json"
$LlamaTag = if ($env:LLAMA_TAG) { $env:LLAMA_TAG } else { "b12189" }   # 검증한 릴리스로 고정
$CudaVer = "12.4"            # Windows 배포 바이너리의 CUDA 버전 → 드라이버 551 이상
$MinDriver = 551
$ModelUrl = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
$ModelSize = 4683074240

function Say($m)  { Write-Host "[w2b] $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "[w2b] 주의: $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "[w2b] 오류: $m" -ForegroundColor Red; exit 1 }
function Ask($q, $d) { $a = Read-Host "  $q [$d]"; if ([string]::IsNullOrWhiteSpace($a)) { $d } else { $a.Trim() } }
function YesNo($q, $d) {
    $hint = if ($d -eq "Y") { "Y/n" } else { "y/N" }
    $a = Read-Host "  $q [$hint]"; if ([string]::IsNullOrWhiteSpace($a)) { $a = $d }
    return $a -match '^[Yy]'
}

New-Item -ItemType Directory -Force -Path (Join-Path $W "dl"), (Join-Path $W "models") | Out-Null
Set-Location $Root

# ------------------------------------------------------------------ 1. 필수 도구
Say "필수 도구 확인"
$Py = $null
foreach ($cand in @(@("py", "-3"), @("python"), @("python3"))) {
    $exe = $cand[0]
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    $rest = @($cand | Select-Object -Skip 1)
    try {
        # Microsoft Store 의 가짜 python.exe 는 아무것도 출력하지 않음 → 버전 확인으로 걸러냄
        $v = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])"
        if ($LASTEXITCODE -eq 0 -and $v -match '^3\.(\d+)$' -and [int]$Matches[1] -ge 8) { $Py = $cand; break }
    } catch {}
}
if (-not $Py) {
    Warn "Python 3.8 이상을 찾지 못했습니다."
    if ((Get-Command winget -ErrorAction SilentlyContinue) -and (YesNo "winget 으로 Python 3.12 를 설치할까요?" "Y")) {
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
        Die "설치가 끝났습니다. 이 창을 닫고 start_laptop_windows.bat 을 다시 실행하세요 (PATH 반영)"
    }
    Die "https://www.python.org/downloads/ 에서 Python 을 설치하세요 ('Add python.exe to PATH' 체크)"
}
$PyExe = $Py[0]; $PyArgs = @($Py | Select-Object -Skip 1)
Say ("Python: " + (& $PyExe @PyArgs --version))

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { Die "curl.exe 가 없습니다 (Windows 10 1803 이상 필요)" }

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $smi) {
    foreach ($p in @("$env:SystemRoot\System32\nvidia-smi.exe", "$env:ProgramFiles\NVIDIA Corporation\NVSMI\nvidia-smi.exe")) {
        if (Test-Path $p) { $env:PATH = (Split-Path $p) + ";" + $env:PATH; $smi = Get-Command nvidia-smi; break }
    }
}
if (-not $smi) { Die "nvidia-smi 가 없습니다. NVIDIA 드라이버(GeForce Experience / NVIDIA App)를 설치하세요" }
$gpuLine = $null
try { $gpuLine = (& nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Select-Object -First 1) } catch {}
if (-not $gpuLine) { Die "NVIDIA GPU 를 찾지 못했습니다 (장치 관리자에서 드라이버 확인)" }
Say "GPU: $gpuLine"
$driver = ($gpuLine -split ",")[1].Trim()
try { if ([int]($driver -split "\.")[0] -lt $MinDriver) { Warn "드라이버 $driver < $MinDriver — CUDA $CudaVer 빌드가 동작하지 않을 수 있습니다. 드라이버를 업데이트하세요" } } catch {}

# ------------------------------------------------------------------ 2. 설정
$cfg = $null
if ((Test-Path $Conf) -and -not $Reconfigure) { $cfg = Get-Content $Conf -Raw -Encoding UTF8 | ConvertFrom-Json }
if (-not $cfg) {
    Say "처음 설정 (Enter = 기본값). 오케스트레이터 PC 의 ./start_orchestrator.sh 가 출력한 주소·토큰을 입력하세요"
    $defId = (($env:COMPUTERNAME -replace '[^A-Za-z0-9]', '').ToUpper())
    if ($defId.Length -gt 10) { $defId = $defId.Substring(0, 10) }
    if (-not $defId) { $defId = "LAP1" }
    $orch = ""
    while (-not $orch) { $orch = Ask "오케스트레이터 주소 (예: http://192.168.0.10:8700)" "" }
    if ($orch -notmatch '^https?://') { $orch = "http://" + $orch }
    if ($orch -notmatch ':\d+/?$') { $orch = $orch.TrimEnd("/") + ":8700" }
    $cfg = [ordered]@{
        orch      = $orch
        token     = Ask "토큰" ""
        id        = Ask "노드 ID (노드마다 다르게, 16자 이내)" $defId
        system_w  = Ask "GPU 를 뺀 나머지 노트북 소비전력(W)" "25"
        solar_kw  = Ask "가상 태양광 용량 kW (비우면 자동)" ""
        gpu_index = "0"
        model_path = ""
        llama_bin  = ""
    }
    ($cfg | ConvertTo-Json) | Set-Content -Path $Conf -Encoding UTF8
    Say "설정 저장: $Conf  (다시 설정: start_laptop_windows.bat -Reconfigure)"
    $cfg = Get-Content $Conf -Raw -Encoding UTF8 | ConvertFrom-Json
}
if (-not $cfg.token) { Warn "토큰이 비어 있습니다 (오케스트레이터를 --token 없이 띄운 경우에만 동작)" }

try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 4 -Uri ($cfg.orch.TrimEnd("/") + "/state") | Out-Null; Say "오케스트레이터 응답 확인: $($cfg.orch)" }
catch { Warn "오케스트레이터($($cfg.orch))에 아직 연결되지 않습니다 — 켜져 있는지, 같은 네트워크인지 확인 (계속 진행)" }

# ------------------------------------------------------------------ 3. llama.cpp (CUDA)
function Fetch($url, $out) {
    $part = "$out.part"
    if ((Test-Path $out) -and -not (Test-Path $part)) { return }
    Say "내려받는 중: $(Split-Path $out -Leaf)"
    & curl.exe -fL --retry 5 --retry-delay 3 -C - --progress-bar -o $part $url
    if ($LASTEXITCODE -ne 0) { Die "다운로드 실패: $url (다시 실행하면 이어받습니다)" }
    Move-Item -Force $part $out
}

$llamaBin = $cfg.llama_bin
if (-not $llamaBin) {
    $llamaDir = Join-Path $W "llama-$LlamaTag-cuda$CudaVer"
    $found = Get-ChildItem -Path $llamaDir -Recurse -Filter "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $found) {
        $base = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag"
        $main = "llama-$LlamaTag-bin-win-cuda-$CudaVer-x64.zip"
        $rt = "cudart-llama-bin-win-cuda-$CudaVer-x64.zip"
        Say "llama.cpp $LlamaTag (CUDA $CudaVer) 준비 — 약 650MB"
        Fetch "$base/$main" (Join-Path $W "dl\$main")
        Fetch "$base/$rt" (Join-Path $W "dl\$rt")
        $tmp = "$llamaDir.tmp"
        if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
        Say "압축 푸는 중"
        try {
            Expand-Archive -Force -Path (Join-Path $W "dl\$main") -DestinationPath $tmp
            Expand-Archive -Force -Path (Join-Path $W "dl\$rt") -DestinationPath "$tmp\_cudart"
        } catch { Die "압축 해제 실패 — .w2b\dl 의 zip 을 지우고 다시 실행하세요 ($_)" }
        $exe = Get-ChildItem -Path $tmp -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        if (-not $exe) { Die "압축 안에서 llama-server.exe 를 찾지 못했습니다" }
        # CUDA 런타임 DLL 을 llama-server.exe 옆으로
        Get-ChildItem -Path "$tmp\_cudart" -Recurse -Filter "*.dll" | Copy-Item -Destination $exe.DirectoryName -Force
        Remove-Item -Recurse -Force "$tmp\_cudart"
        if (Test-Path $llamaDir) { Remove-Item -Recurse -Force $llamaDir }
        Move-Item $tmp $llamaDir
        $found = Get-ChildItem -Path $llamaDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
        Say "llama-server 설치: $($found.FullName)"
    }
    $llamaBin = $found.FullName
}
if (-not (Test-Path $llamaBin)) { Die "llama-server 실행 파일이 없습니다: $llamaBin" }

# ------------------------------------------------------------------ 4. 모델
$modelPath = $cfg.model_path
if (-not $modelPath) {
    $modelPath = Join-Path $W ("models\" + ($ModelUrl -split "/")[-1])
    if (-not (Test-Path $modelPath) -or (Test-Path "$modelPath.part")) {
        $drive = (Get-Item $W).PSDrive
        if ($drive.Free -and $drive.Free -lt ($ModelSize + 500MB)) { Die "디스크 공간 부족: 모델에 약 4.7GB 필요" }
        Say "모델 내려받기 (Qwen2.5-7B-Instruct Q4_K_M, 4.7GB)"
        Fetch $ModelUrl $modelPath
    }
    $sz = (Get-Item $modelPath).Length
    if ($sz -ne $ModelSize) { Die "모델 크기 불일치 ($sz / $ModelSize 바이트) — $modelPath 를 지우고 다시 실행" }
}
if (-not (Test-Path $modelPath)) { Die "모델 파일이 없습니다: $modelPath" }

# ------------------------------------------------------------------ 5. 노트북 전원
try {
    Add-Type -AssemblyName System.Windows.Forms
    if ([System.Windows.Forms.SystemInformation]::PowerStatus.PowerLineStatus -ne "Online") {
        Warn "충전기가 연결되지 않았습니다 — 배터리 구동 시 GPU 전력이 크게 제한됩니다"
    }
} catch {}
Say "Windows 설정 > 시스템 > 전원 에서 전원 모드를 '최고 성능' 으로 두세요 (제조사 앱이 있으면 '성능' 모드)"

# ------------------------------------------------------------------ 6. 점검 → 실행
# 외부 프로그램의 stderr 가 스크립트를 멈추지 않도록 (Windows PowerShell 5.1)
$ErrorActionPreference = "Continue"
$agent = Join-Path $Root "agent.py"
$argv = @($agent, "--orch", $cfg.orch, "--id", $cfg.id, "--system-w", "$($cfg.system_w)",
          "--gpu-index", "$($cfg.gpu_index)", "--llama-bin", $llamaBin, "--gguf", $modelPath)
if ($cfg.token)    { $argv += @("--token", $cfg.token) }
if ($cfg.solar_kw) { $argv += @("--solar-kw", "$($cfg.solar_kw)") }

if (-not $SkipCheck) {
    Say "점검 실행 (모델 로딩 때문에 수십 초 걸릴 수 있음)"
    & $PyExe @PyArgs @argv --check
    if ($LASTEXITCODE -ne 0) {
        if ($CheckOnly) { exit 1 }
        if (-not (YesNo "점검에 실패 항목이 있습니다. 그래도 시작할까요?" "N")) { exit 1 }
    }
}
if ($CheckOnly) { exit 0 }

Say "노드 $($cfg.id) 시작 → $($cfg.orch)   (종료: Ctrl+C 또는 창 닫기)"
while ($true) {
    & $PyExe @PyArgs @argv
    if ($LASTEXITCODE -eq 0) { break }
    Warn "에이전트 종료(코드 $LASTEXITCODE) — 5초 뒤 재시작"
    Start-Sleep -Seconds 5
}
