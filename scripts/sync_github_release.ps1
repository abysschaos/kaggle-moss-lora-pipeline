<#
.SYNOPSIS
    同步 GitHub 仓库主页元数据（Description / Homepage / Topics）与 Releases 发行页及构建资产。
.DESCRIPTION
    完全兼容 Windows PowerShell 5.1+ 且零第三方 CLI 依赖。
    优先读取环境变量 $env:GITHUB_TOKEN，若未配置则从本地 `git credential fill` 内存读取 GitHub 凭证（全程绝不输出或落盘密钥）。
    所有含中文的 JSON Payload 均严格转换为 UTF-8 字节流发送，彻底避免 PowerShell 5.1 默认编码乱码。
.PARAMETER Repo
    目标 GitHub 仓库（owner/repo），默认 abysschaos/kaggle-moss-lora-pipeline。
.PARAMETER SyncMetadata
    同步更新仓库 Description、Homepage 与 Topics。
.PARAMETER Tag
    目标 Release 对应的 Git Tag（例如 v1.2.2）。
.PARAMETER Title
    GitHub Release 展示标题。
.PARAMETER NotesFile
    Markdown 格式的 Release 正文文件路径（UTF-8）。
.PARAMETER Assets
    待挂载上传至该 Release 的本地文件路径数组（例如 MOSS_LoRA_Trainer.exe、SHA256SUMS.txt）。
.PARAMETER MakeLatest
    控制 GitHub 将该 Release 标记为 Latest（true / false / legacy）。
#>
param(
    [string]$Repo = "abysschaos/kaggle-moss-lora-pipeline",
    [switch]$SyncMetadata,
    [string]$Tag = "",
    [string]$Title = "",
    [string]$NotesFile = "",
    [string[]]$Assets = @(),
    [ValidateSet("true", "false", "legacy")]
    [string]$MakeLatest = "true"
)

$ErrorActionPreference = "Stop"

# 确保 TLS 1.2 在 PowerShell 5.1 下默认开启
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch { }

function Get-GitHubToken {
    if ($env:GITHUB_TOKEN -and $env:GITHUB_TOKEN.Trim()) {
        return $env:GITHUB_TOKEN.Trim()
    }
    $credPrompt = "protocol=https`nhost=github.com`n`n"
    $credLines = $credPrompt | & git credential fill 2>$null
    foreach ($line in $credLines) {
        if ($line -match '^password=(.+)$') {
            return $Matches[1].Trim()
        }
    }
    throw "未能获取 GitHub 访问令牌。请设置环境变量 GITHUB_TOKEN 或确认 git credential 已登录 github.com。"
}

function Invoke-GitHubJson {
    param(
        [string]$Method,
        [string]$Uri,
        [hashtable]$Headers,
        [object]$BodyObject = $null
    )
    if ($null -ne $BodyObject) {
        $jsonText = $BodyObject | ConvertTo-Json -Depth 6
        $utf8Bytes = [System.Text.Encoding]::UTF8.GetBytes($jsonText)
        return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers -Body $utf8Bytes -ContentType "application/json; charset=utf-8"
    }
    return Invoke-RestMethod -Method $Method -Uri $Uri -Headers $Headers
}

function Get-HttpStatusCode {
    param($ExceptionRecord)
    if ($null -eq $ExceptionRecord.Exception.Response) {
        return 0
    }
    try {
        return [int]$ExceptionRecord.Exception.Response.StatusCode
    } catch {
        return 0
    }
}

$token = Get-GitHubToken
$authHeaders = @{
    "User-Agent"           = "kaggle-moss-lora-pipeline-sync"
    "Accept"               = "application/vnd.github+json"
    "Authorization"        = "Bearer $token"
    "X-GitHub-Api-Version" = "2022-11-28"
}
$token = $null

# 1. 同步 GitHub 仓库主页元数据与 Topics
if ($SyncMetadata) {
    Write-Host "[1/2] 正在同步仓库主页元数据 ($Repo)..." -ForegroundColor Cyan
    $metaPayload = @{
        description = "OpenMOSS MOSS-Image (5B DiT) 本地消费级显卡 (6G~16G) 与 Kaggle 云端双轨 LoRA/DoRA/LoHa/LoKr 自动化训练管线（零黑框 GUI / 多子集配比 / 实时看板 / 暂停续炼）"
        homepage    = "https://github.com/$Repo/releases/latest"
    }
    $updatedRepo = Invoke-GitHubJson -Method "PATCH" -Uri "https://api.github.com/repos/$Repo" -Headers $authHeaders -BodyObject $metaPayload
    Write-Host "      Description 与 Homepage 已同步: $($updatedRepo.homepage)" -ForegroundColor Green

    $topicsPayload = @{
        names = @(
            "moss-image",
            "openmoss",
            "lora-training",
            "dit",
            "flow-matching",
            "dora",
            "lycoris",
            "comfyui",
            "kaggle",
            "windows",
            "kohakumosstrainer"
        )
    }
    $updatedTopics = Invoke-GitHubJson -Method "PUT" -Uri "https://api.github.com/repos/$Repo/topics" -Headers $authHeaders -BodyObject $topicsPayload
    Write-Host ("      Topics 已同步 ({0} 个): {1}" -f $updatedTopics.names.Count, ($updatedTopics.names -join ", ")) -ForegroundColor Green
}

# 2. 幂等创建/更新 GitHub Release 并上传附件
if ($Tag -and $Tag.Trim()) {
    $cleanTag = $Tag.Trim()
    $releaseTitle = if ($Title -and $Title.Trim()) { $Title.Trim() } else { $cleanTag }
    $releaseBody = ""
    if ($NotesFile -and $NotesFile.Trim()) {
        if (-not (Test-Path $NotesFile)) {
            throw "未找到发行说明文件: $NotesFile"
        }
        $releaseBody = [System.IO.File]::ReadAllText((Resolve-Path $NotesFile).Path, (New-Object System.Text.UTF8Encoding($false)))
    }

    Write-Host "[Release] 正在同步发行页: $cleanTag (make_latest=$MakeLatest)..." -ForegroundColor Cyan
    $encodedTag = [System.Uri]::EscapeDataString($cleanTag)
    $existingRelease = $null
    try {
        $existingRelease = Invoke-GitHubJson -Method "GET" -Uri "https://api.github.com/repos/$Repo/releases/tags/$encodedTag" -Headers $authHeaders
    } catch {
        $code = Get-HttpStatusCode $_
        if ($code -ne 404) {
            throw
        }
    }

    $releasePayload = @{
        tag_name   = $cleanTag
        name       = $releaseTitle
        body       = $releaseBody
        draft      = $false
        prerelease = $false
        make_latest = $MakeLatest
    }

    if ($null -ne $existingRelease) {
        $targetRelease = Invoke-GitHubJson -Method "PATCH" -Uri "https://api.github.com/repos/$Repo/releases/$($existingRelease.id)" -Headers $authHeaders -BodyObject $releasePayload
        Write-Host "      已更新现有 Release #$($targetRelease.id): $($targetRelease.html_url)" -ForegroundColor Green
    } else {
        $releasePayload["target_commitish"] = "main"
        $targetRelease = Invoke-GitHubJson -Method "POST" -Uri "https://api.github.com/repos/$Repo/releases" -Headers $authHeaders -BodyObject $releasePayload
        Write-Host "      已创建全新 Release #$($targetRelease.id): $($targetRelease.html_url)" -ForegroundColor Green
    }

    if ($Assets -and $Assets.Count -gt 0) {
        $existingAssets = @()
        if ($targetRelease.assets) {
            $existingAssets = @($targetRelease.assets)
        }
        foreach ($assetPath in $Assets) {
            if (-not $assetPath -or -not (Test-Path $assetPath)) {
                throw "待上传构建产物不存在: $assetPath"
            }
            $resolvedPath = (Resolve-Path $assetPath).Path
            $fileName = [System.IO.Path]::GetFileName($resolvedPath)

            foreach ($oldAsset in $existingAssets) {
                if ($oldAsset.name -eq $fileName) {
                    Write-Host "      移除同名旧附件: $fileName (#$($oldAsset.id))" -ForegroundColor Yellow
                    Invoke-GitHubJson -Method "DELETE" -Uri "https://api.github.com/repos/$Repo/releases/assets/$($oldAsset.id)" -Headers $authHeaders | Out-Null
                }
            }

            $ext = [System.IO.Path]::GetExtension($fileName).ToLowerInvariant()
            $contentType = "application/octet-stream"
            if ($ext -eq ".txt" -or $ext -eq ".md") {
                $contentType = "text/plain; charset=utf-8"
            } elseif ($ext -eq ".zip") {
                $contentType = "application/zip"
            }

            $fileBytes = [System.IO.File]::ReadAllBytes($resolvedPath)
            $escapedName = [System.Uri]::EscapeDataString($fileName)
            $uploadUri = "https://uploads.github.com/repos/$Repo/releases/$($targetRelease.id)/assets?name=$escapedName"
            $uploaded = Invoke-RestMethod -Method "POST" -Uri $uploadUri -Headers $authHeaders -Body $fileBytes -ContentType $contentType
            Write-Host ("      已挂载产物: {0} ({1} bytes)" -f $uploaded.name, $uploaded.size) -ForegroundColor Green
        }
    }
}

Write-Host "[完成] GitHub 同步执行完毕。" -ForegroundColor Green
