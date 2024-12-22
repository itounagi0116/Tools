# 無効なPathと重複したPathを除外して保存するスクリプト

# 現在のPath環境変数を取得
$pathEntries = $env:Path -split ';'

# 重複と無効なパスを除外する
$cleanedPath = $pathEntries |
    ForEach-Object {
        # 空白とトリム済みのパス
        $_.Trim()
    } |
    Where-Object {
        # パスが空でなく、存在するものを残す
        ($_ -ne "") -and (Test-Path $_)
    } |
    Select-Object -Unique

# クリーンなPathを再構築
$updatedPath = $cleanedPath -join ';'

# 更新されたPathを設定（ユーザ環境変数）
[System.Environment]::SetEnvironmentVariable("Path", $updatedPath, "User")

# 確認のため出力
Write-Host "Updated Path:" -ForegroundColor Green
$cleanedPath | ForEach-Object { Write-Host $_ }

# 確認用にログファイルに保存
$logFile = "Path_Cleanup_Log.txt"
$cleanedPath | Out-File -FilePath $logFile -Encoding UTF8
Write-Host "Path cleanup log saved to $logFile" -ForegroundColor Cyan
