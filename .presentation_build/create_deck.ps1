$ErrorActionPreference = 'Stop'
$root = (Get-Location).Path
$assets = Join-Path $root '.presentation_build\assets'
$outDir = Join-Path $root '.presentation_output'
$pngDir = Join-Path $root '.presentation_build\rendered'
New-Item -ItemType Directory -Force -Path $outDir,$pngDir | Out-Null
$pptxPath = Join-Path $outDir 'WattWise_AI_Project_Presentation_EN.pptx'
if (Test-Path $pptxPath) { Remove-Item -LiteralPath $pptxPath }

# Office PowerPoint is used here because the presentation artifact runtime is not
# available in this environment; all slide text, tables, diagrams and bars are editable.
$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$ppt.WindowState = 2
$pres = $ppt.Presentations.Add()
$pres.PageSetup.SlideSize = 16
$pres.PageSetup.SlideWidth = 960
$pres.PageSetup.SlideHeight = 540
$W = [double]$pres.PageSetup.SlideWidth
$H = [double]$pres.PageSetup.SlideHeight

function RGB([int]$r,[int]$g,[int]$b) { return [int]($r + ($g * 256) + ($b * 65536)) }
$C = @{
  bg = (RGB 11 23 32)       # midnight navy
  bg2 = (RGB 18 40 50)     # lifted navy
  bg3 = (RGB 27 53 64)     # surface
  teal = (RGB 70 199 184)  # teal
  mint = (RGB 160 231 215) # mint
  amber = (RGB 227 170 99) # warm amber
  white = (RGB 241 246 242)
  soft = (RGB 173 192 197)
  muted = (RGB 115 135 140)
  border = (RGB 57 78 88)
  dark = (RGB 16 32 30)
  tableFill = (RGB 26 43 53)
}

function Set-TextStyle($shape,[string]$font='Arial',[double]$size=16,[int]$color=$C.white,[bool]$bold=$false,[int]$align=1,[bool]$rtl=$false) {
  $tf = $shape.TextFrame2
  $tf.MarginLeft = 2; $tf.MarginRight = 2; $tf.MarginTop = 1; $tf.MarginBottom = 1
  $tf.WordWrap = -1; $tf.AutoSize = 0; $tf.VerticalAnchor = 3
  $tr = $tf.TextRange
  $tr.Font.Name = $font; $tr.Font.Size = $size; $tr.Font.Bold = $bold
  $tr.Font.Fill.ForeColor.RGB = $color
  $tr.ParagraphFormat.Alignment = $align
  if ($rtl) { try { $tr.ParagraphFormat.TextDirection = 2 } catch {} }
}
function Add-Text($slide,[string]$text,[double]$x,[double]$y,[double]$w,[double]$h,[double]$size=16,[int]$color=$C.white,[bool]$bold=$false,[int]$align=1,[bool]$rtl=$false,[string]$font='Arial') {
  $s = $slide.Shapes.AddTextbox(1,$x,$y,$w,$h)
  $s.Fill.Transparency = 1; $s.Line.Visible = 0
  $s.TextFrame2.TextRange.Text = $text
  Set-TextStyle $s $font $size $color $bold $align $rtl
  return $s
}
function Add-Rect($slide,[double]$x,[double]$y,[double]$w,[double]$h,[int]$fill,[int]$line=-1,[double]$lineWidth=0,[int]$geometry=1) {
  $s = $slide.Shapes.AddShape($geometry,$x,$y,$w,$h)
  $s.Fill.Solid(); $s.Fill.ForeColor.RGB = $fill
  if ($line -lt 0) { $s.Line.Visible = 0 } else { $s.Line.ForeColor.RGB = $line; $s.Line.Weight = $lineWidth }
  return $s
}
function Add-Line($slide,[double]$x1,[double]$y1,[double]$x2,[double]$y2,[int]$color=$C.border,[double]$width=1,[bool]$arrow=$false) {
  $s = $slide.Shapes.AddConnector(1,$x1,$y1,$x2,$y2)
  $s.Line.ForeColor.RGB = $color; $s.Line.Weight = $width
  if ($arrow) { $s.Line.EndArrowheadStyle = 3 }
  return $s
}
function Add-Image($slide,[string]$path,[double]$x,[double]$y,[double]$w,[double]$h) {
  return $slide.Shapes.AddPicture($path,0,-1,$x,$y,$w,$h)
}
function Add-Notes($slide,[string]$text) {
  try { $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $text } catch {}
}
function New-Slide([string]$title,[int]$n,[string]$section='WattWise AI') {
  $s = $pres.Slides.Add($pres.Slides.Count+1,12)
  $s.FollowMasterBackground = $false
  $s.Background.Fill.Solid(); $s.Background.Fill.ForeColor.RGB = $C.bg
  Add-Rect $s 42 37 5 37 $C.teal | Out-Null
  Add-Text $s $title 66 32 850 48 27 $C.white $true 3 $true | Out-Null
  Add-Line $s 42 88 918 88 $C.border 0.8 | Out-Null
  Add-Text $s 'WattWise AI  |  Advanced ML' 42 512 780 15 8.5 $C.muted $false 1 $false | Out-Null
  Add-Text $s ("{0:D2}" -f $n) 875 510 42 18 9 $C.teal $true 2 $false | Out-Null
  return $s
}
function Add-Node($slide,[string]$label,[double]$x,[double]$y,[double]$w,[double]$h,[int]$accent=$C.teal) {
  $n = Add-Rect $slide $x $y $w $h $C.bg2 $accent 1.2 5
  $n.TextFrame2.TextRange.Text = $label
  Set-TextStyle $n 'Arial' 13.5 $C.white $true 2 $true
  return $n
}
function Add-Table($slide,[string[]]$headers,[object[]]$rows,[double]$x,[double]$y,[double]$w,[double]$h,[double[]]$widths,[double]$fontSize=12,[bool]$rtlHeaders=$true) {
  $rn = $rows.Count + 1; $cn = $headers.Count
  $shape = $slide.Shapes.AddTable($rn,$cn,$x,$y,$w,$h)
  $tbl = $shape.Table
  if ($widths.Count -eq $cn) { for($col=0;$col -lt $cn;$col++){ $tbl.Columns.Item($col+1).Width = $widths[$col] } }
  for($r=1;$r -le $rn;$r++) {
    $tbl.Rows.Item($r).Height = $h / $rn
    for($col=1;$col -le $cn;$col++) {
      $cell = $tbl.Cell($r,$col); $sh = $cell.Shape
      $sh.Fill.Solid()
      $sh.Fill.Solid()
      if($r -eq 1){ $sh.Fill.ForeColor.RGB = $C.bg3 } elseif($r % 2 -eq 0){ $sh.Fill.ForeColor.RGB = $C.bg2 } else { $sh.Fill.ForeColor.RGB = $C.tableFill }
      $sh.TextFrame2.MarginLeft = 6; $sh.TextFrame2.MarginRight = 6; $sh.TextFrame2.MarginTop = 2; $sh.TextFrame2.MarginBottom = 2
      $sh.TextFrame2.VerticalAnchor = 3
      if($r -eq 1){ $v = [string]$headers[$col-1] } else { $v = [string]$rows[$r-2][$col-1] }
      $sh.TextFrame2.TextRange.Text = $v
      $tr = $sh.TextFrame2.TextRange
      $tr.Font.Name = 'Arial'; $tr.Font.Size = $(if($r -eq 1){$fontSize}else{$fontSize}); $tr.Font.Bold = ($r -eq 1)
      $tr.Font.Fill.ForeColor.RGB = $(if($r -eq 1){$C.mint}else{$C.white})
      $tr.ParagraphFormat.Alignment = $(if($r -eq 1 -and $rtlHeaders){3}elseif($col -eq 1){1}else{2})
      if($r -eq 1 -and $rtlHeaders){ try { $tr.ParagraphFormat.TextDirection = 2 } catch {} }
    }
  }
  return $shape
}
function Add-Bar($slide,[string]$label,[double]$value,[double]$max,[double]$x,[double]$y,[double]$labelW=180,[double]$barW=500,[int]$color=$C.teal) {
  $isEnglish = $label -match '[A-Za-z]'
  Add-Text $slide $label $x $y $labelW 22 13 $C.soft $false $(if($isEnglish){1}else{3}) (-not $isEnglish) | Out-Null
  Add-Rect $slide ($x+$labelW+14) ($y+5) $barW 12 $C.bg2 | Out-Null
  $fillW = [Math]::Max(2,$barW * $value / $max)
  Add-Rect $slide ($x+$labelW+14) ($y+5) $fillW 12 $color | Out-Null
  Add-Text $slide ([string]::Format('{0:N1}', $value)) ($x+$labelW+$barW+26) ($y-2) 70 24 12 $C.white $true 1 $false | Out-Null
}

# 1 — Cover
$s = $pres.Slides.Add(1,12); $s.FollowMasterBackground = $false; $s.Background.Fill.Solid(); $s.Background.Fill.ForeColor.RGB = $C.bg
Add-Image $s (Join-Path $assets 'wattwise_cover.png') 0 0 $W $H | Out-Null
Add-Rect $s 0 0 480 $H $C.bg | Out-Null
# a softly layered dark area keeps the cover title readable without obstructing the illustration
$s.Shapes.Item($s.Shapes.Count).Fill.Transparency = 0.12
Add-Text $s 'WattWise AI' 52 128 425 68 39 $C.white $true 1 $false 'Arial' | Out-Null
Add-Rect $s 54 207 66 4 $C.teal | Out-Null
Add-Text $s "Energy Forecasting`n& Anomaly Detection" 52 235 420 94 24 $C.white $true 3 $true | Out-Null
Add-Text $s 'Advanced Machine Learning Project' 52 354 420 28 17 $C.mint $true 3 $true | Out-Null
Add-Text $s 'Short-horizon forecasts • Explainable signals • Human review' 52 395 420 45 14 $C.soft $false 3 $true | Out-Null
Add-Text $s 'Advanced Machine Learning  |  2026' 52 486 420 22 11 $C.soft $false 1 $false | Out-Null
Add-Notes $s 'WattWise AI project presentation. The cover is a conceptual illustration, not a photograph of a real home or appliance.'

# 2 — Team
$s = New-Slide 'Project Team' 2 'Project and Team'
Add-Text $s 'Team Members' 570 119 330 32 19 $C.mint $true 3 $true | Out-Null
$team = @(
  @('Abdelrhman Mohamed Ahmed Al Ghonimi','223101228','Team Member'),
  @('Denar Reda Roushdy Elsayed','223101171','Team Member'),
  @('Mariem Abd-Elhamied Abo-Shaeshaa','223101341','Team Member'),
  @('Shrouk Ashraf Seria','223101203','Team Member')
)
Add-Table $s @('Name','Student ID','Role') $team 54 164 852 190 @(510,190,152) 13.5 $true | Out-Null
Add-Text $s 'Course' 675 390 220 24 14 $C.soft $false 3 $true | Out-Null
Add-Text $s 'Advanced Machine Learning' 520 422 376 32 19 $C.white $true 3 $false | Out-Null
Add-Notes $s 'Team member names and student IDs were provided by the project team.'

# 3 — Problem & use
$s = New-Slide 'Problem and Scope' 3 'Project Definition'
Add-Text $s 'Question Addressed' 500 119 405 30 19 $C.mint $true 3 $true | Out-Null
Add-Text $s 'How much energy will household appliances use in the next interval, and how far might observed use depart from the forecast?' 125 160 780 74 22 $C.white $true 3 $true | Out-Null
$steps = @(
  @('Household Readings','Available sensor and energy readings at time t'),
  @('Next-Interval Forecast','Observed energy at the next row'),
  @('Residual Analysis','Actual − predicted'),
  @('Human Review','A signal worth investigating')
)
$sx=56; $sw=190; $gap=39
for($i=0;$i -lt $steps.Count;$i++){
  $x=$sx+$i*($sw+$gap); Add-Node $s $steps[$i][0] $x 280 $sw 50 $(if($i -eq 3){$C.amber}else{$C.teal}) | Out-Null
  Add-Text $s $steps[$i][1] $x 341 $sw 46 11.5 $C.soft $false 2 $true | Out-Null
  if($i -lt 3){ Add-Line $s ($x+$sw+4) 305 ($x+$sw+$gap-5) 305 $C.mint 1.5 $true | Out-Null }
}
Add-Text $s 'A residual does not diagnose waste or a fault. The UCI dataset has no verified waste or fault labels.' 105 435 790 42 15 $C.amber $true 3 $true | Out-Null

# 4 — Data
$s = New-Slide 'Dataset and Source' 4 'Data'
Add-Text $s 'UCI Appliances Energy Prediction' 500 116 400 30 18 $C.mint $true 3 $false | Out-Null
$dataRows = @(
  @('Observations','19,735'),
  @('Original columns','29'),
  @('Sampling interval','10 minutes'),
  @('Coverage','Jan–May 2016'),
  @('Missing values / duplicates','0 / 0'),
  @('Target','Appliance energy (Wh per interval)')
)
Add-Table $s @('Field','Value') $dataRows 72 162 816 250 @(455,361) 14 $true | Out-Null
Add-Text $s 'UCI Machine Learning Repository' 72 427 816 18 10.5 $C.soft $false 1 $false | Out-Null
Add-Text $s 'archive.ics.uci.edu/dataset/374/appliances+energy+prediction' 72 448 816 18 9.5 $C.soft $false 1 $false | Out-Null
Add-Notes $s 'Source: UCI Machine Learning Repository, Appliances Energy Prediction. Candanedo, Feldheim & Deramaix (2017), Energy and Buildings, DOI: 10.1016/j.enbuild.2017.01.083. Dataset facts are from the inspected source CSV and reports.'

# 5 — Pipeline diagram
$s = New-Slide 'Data-to-Forecast Pipeline' 5 'Methodology'
$nodes = @("UCI Dataset`n19,735 rows","Validate & Clean`nTimestamps","Feature Engineering`n63 features","Chronological Split`n70 / 15 / 15","Select on`nvalidation only","Held-out Test`nthen review")
$nx=42; $nw=130; $ng=20
for($i=0;$i -lt $nodes.Count;$i++){
  $x=$nx+$i*($nw+$ng); Add-Node $s $nodes[$i] $x 204 $nw 82 $(if($i -eq 4){$C.amber}else{$C.teal}) | Out-Null
  if($i -lt $nodes.Count-1){Add-Line $s ($x+$nw+2) 245 ($x+$nw+$ng-3) 245 $C.mint 1.4 $true | Out-Null}
}
Add-Line $s 67 336 893 336 $C.border 0.8 | Out-Null
Add-Text $s 'Forecast Inputs' 680 360 212 24 14 $C.mint $true 3 $true | Out-Null
Add-Text $s 'At time t: sensor readings, current energy use, and available historical features.' 70 394 820 30 16 $C.white $false 3 $true | Out-Null
Add-Text $s 'Target: appliance energy observed in the next row, usually 10 minutes later.' 70 434 820 30 16 $C.soft $false 3 $true | Out-Null

# 6 — Features and leakage safeguards
$s = New-Slide 'Feature Engineering and Leakage Controls' 6 'Data and Methodology'
Add-Text $s '63 features in the recorded experiment' 570 116 330 30 18 $C.mint $true 3 $true | Out-Null
$featureRows = @(
  @('Current appliance use','appliances_current'),
  @('Consumption lags','1, 2, 3, 6, 12, 144 samples'),
  @('Shifted rolling summaries','Mean and standard deviation over 3, 6, 12, 144 samples'),
  @('Recent changes','Energy, lighting, and selected temperature/humidity readings'),
  @('Cyclical time','Hour, day, daily and weekly cyclical encodings')
)
Add-Table $s @('Feature group','Examples') $featureRows 58 157 844 221 @(300,544) 12.7 $true | Out-Null
Add-Text $s 'Leakage Controls' 650 399 240 25 16 $C.mint $true 3 $true | Out-Null
Add-Text $s "• Forward-fill from past values only; never back-fill from the future.`n• Rolling windows are shifted and exclude the next-step target.`n• Chronological splits use disjoint target timestamps." 80 432 810 60 13 $C.soft $false 3 $true | Out-Null

# 7 — Split and model families
$s = New-Slide 'Temporal Evaluation and Candidate Models' 7 'Training and Evaluation'
Add-Text $s 'Chronological Split' 690 115 210 26 17 $C.mint $true 3 $true | Out-Null
$baseX=68; $baseY=164; $baseW=824; $h=50
$widthTrain=$baseW*0.70; $widthVal=$baseW*0.15; $widthTest=$baseW*0.15
Add-Rect $s $baseX $baseY $widthTrain $h $C.teal | Out-Null
Add-Rect $s ($baseX+$widthTrain) $baseY $widthVal $h $C.amber | Out-Null
Add-Rect $s ($baseX+$widthTrain+$widthVal) $baseY $widthTest $h $C.mint | Out-Null
Add-Text $s 'Train 70%  |  n=13,713' ($baseX+10) ($baseY+10) ($widthTrain-20) 30 14 $C.dark $true 2 $false | Out-Null
Add-Text $s 'Validation 15%' ($baseX+$widthTrain) ($baseY+10) $widthVal 30 9 $C.dark $true 2 $false | Out-Null
Add-Text $s 'Test 15%' ($baseX+$widthTrain+$widthVal) ($baseY+10) $widthTest 30 9 $C.dark $true 2 $false | Out-Null
Add-Text $s 'Train 13,713  |  Validation 2,938  |  Test 2,939' 170 229 620 24 12.5 $C.soft $false 2 $false | Out-Null
Add-Line $s 65 275 895 275 $C.border 0.8 | Out-Null
Add-Text $s 'Candidate Models' 650 295 245 27 17 $C.mint $true 3 $true | Out-Null
Add-Text $s "Baselines: Persistence, Dummy, Ridge`nForests: Random Forest, Extra Trees`nBoosting: HistGradientBoosting, XGBoost`nEnsembles: Voting (top 3), Temporal Stacking" 420 337 470 118 13 $C.white $false 1 $false | Out-Null
Add-Text $s 'The validation set selects the model and residual threshold; test data do not enter selection.' 58 467 842 27 14 $C.amber $true 3 $true | Out-Null

# 8 — CV, tuning, ensembles
$s = New-Slide 'Tuning and Ensemble Models' 8 'Training and Evaluation'
Add-Text $s 'Expanding-Window Validation' 665 116 235 25 17 $C.mint $true 3 $true | Out-Null
Add-Node $s "Base Models`n3 expanding folds" 650 158 245 62 $C.teal | Out-Null
Add-Node $s "Tuned Model`n3 outer × 3 inner folds" 355 158 245 62 $C.amber | Out-Null
Add-Node $s "Optuna`nMAE tuning inside training" 60 158 245 62 $C.teal | Out-Null
Add-Line $s 650 189 603 189 $C.mint 1.3 $true | Out-Null
Add-Line $s 355 189 308 189 $C.mint 1.3 $true | Out-Null
Add-Text $s 'Ensembles' 790 261 105 25 17 $C.mint $true 3 $true | Out-Null
Add-Text $s "Voting: equal-weight average of the three models with the lowest validation error.`nTemporal stacking: out-of-fold forecasts train a linear meta-learner." 90 299 800 62 15 $C.white $false 3 $true | Out-Null
Add-Line $s 66 382 894 382 $C.border 0.8 | Out-Null
Add-Text $s 'Before Final Comparison' 690 400 205 23 15 $C.mint $true 3 $true | Out-Null
Add-Text $s 'Candidates were refit on the same training and validation data, then scored on identical test rows. Ensemble CV is omitted because ensembles were selected using validation.' 90 432 800 48 14 $C.soft $false 3 $true | Out-Null

# 9 — validation selection
$s = New-Slide 'Model Selection on Validation' 9 'Results'
Add-Text $s 'Selection metric: minimum validation MAE' 470 111 430 28 15 $C.mint $true 1 $false | Out-Null
$valRows = @(
  @('Voting (top 3)  ✓','21.939','56.356','0.610'),
  @('Tuned HistGradientBoosting','21.989','56.821','0.603'),
  @('HistGradientBoosting','22.121','57.325','0.596'),
  @('XGBoost','22.335','55.860','0.617')
)
Add-Table $s @('Model','MAE · Wh','RMSE · Wh','R²') $valRows 69 157 822 170 @(382,160,160,120) 13 $true | Out-Null
Add-Text $s 'Selected Model' 700 365 190 24 15 $C.soft $false 3 $true | Out-Null
Add-Text $s 'Voting (top 3)' 482 397 410 40 27 $C.mint $true 3 $false | Out-Null
Add-Text $s 'Equal weights: Tuned HistGradientBoosting  |  HistGradientBoosting  |  XGBoost' 110 455 780 25 11.5 $C.soft $false 1 $false | Out-Null

# 10 — held-out test comparison
$s = New-Slide 'Held-Out Test Model Comparison' 10 'Results'
Add-Text $s 'n = 2,939  |  MAE / RMSE: Wh per 10 min  |  R²: unitless' 335 102 565 22 10.5 $C.soft $false 1 $false | Out-Null
$testRows = @(
  @('Tuned HistGradientBoosting','22.268','58.872','0.581',''),
  @('Voting (top 3)','22.385','58.723','0.583','Selected'),
  @('HistGradientBoosting','22.662','59.814','0.568',''),
  @('XGBoost','22.820','58.326','0.589',''),
  @('Temporal Stacking','23.722','57.941','0.594',''),
  @('Extra Trees','26.159','57.324','0.603',''),
  @('HistGradientBoosting (squared)','26.644','57.693','0.598',''),
  @('Persistence','26.723','66.786','0.461',''),
  @('Ridge','27.878','59.990','0.565',''),
  @('Random Forest','28.987','58.660','0.584',''),
  @('Dummy mean','52.354','90.977','-0.000','')
)
Add-Table $s @('Model','MAE','RMSE','R²','Selection') $testRows 46 132 868 319 @(390,135,135,95,113) 11 $true | Out-Null
Add-Text $s 'Retrospective test leaders — MAE: Tuned HistGradientBoosting  |  RMSE: Extra Trees' 76 461 820 24 10.5 $C.amber $true 1 $false | Out-Null

# 11 — temporal CV
$s = New-Slide 'Temporal Cross-Validation Results' 11 'Results'
Add-Text $s '3 expanding folds  |  mean ± population standard deviation' 474 102 425 22 10 $C.soft $false 1 $false | Out-Null
$cvRows = @(
  @('Persistence','30.387 ± 1.021','73.219 ± 1.493','0.480 ± 0.027'),
  @('Dummy mean','60.099 ± 1.275','101.743 ± 4.454','-0.001 ± 0.001'),
  @('Ridge','36.608 ± 8.038','69.480 ± 2.314','0.527 ± 0.075'),
  @('Random Forest','31.720 ± 1.812','64.880 ± 0.728','0.590 ± 0.046'),
  @('Extra Trees','30.113 ± 1.907','64.122 ± 0.348','0.600 ± 0.040'),
  @('HistGradientBoosting','27.049 ± 1.022','67.234 ± 1.177','0.561 ± 0.027'),
  @('HistGradientBoosting (squared)','29.386 ± 2.179','64.941 ± 1.546','0.588 ± 0.057'),
  @('XGBoost','27.083 ± 1.461','65.273 ± 0.643','0.586 ± 0.032'),
  @('Tuned HGB · nested','27.093 ± 1.159','67.308 ± 1.296','0.560 ± 0.029')
)
Add-Table $s @('Model','CV MAE · Wh','CV RMSE · Wh','CV R²') $cvRows 49 131 862 322 @(332,190,200,140) 10.6 $true | Out-Null
Add-Text $s 'Nested tuning: 3 outer × 3 inner folds. No CV score for validation-composed ensembles.' 75 466 820 21 9.5 $C.soft $false 1 $false | Out-Null

# 12 — Residual diagnostics
$s = New-Slide 'Residual Error Diagnostics' 12 'Results'
Add-Text $s 'Selected model | held-out test' 548 112 350 25 14 $C.mint $true 1 $false | Out-Null
Add-Text $s 'MAE  22.385 Wh     RMSE  58.723 Wh     R²  0.583' 190 150 710 34 20 $C.white $true 2 $false | Out-Null
Add-Line $s 65 203 895 203 $C.border 0.8 | Out-Null
Add-Text $s 'MAE by time block / demand range' 550 221 345 25 14 $C.mint $true 1 $false | Out-Null
Add-Bar $s 'Time block 1' 20.4 90 65 267 150 545 $C.teal
Add-Bar $s 'Time block 2' 18.2 90 65 305 150 545 $C.teal
Add-Bar $s 'Time block 3' 28.6 90 65 343 150 545 $C.teal
Add-Bar $s 'High demand >110 Wh' 86.4 90 65 381 150 545 $C.amber
Add-Text $s 'Absolute-Error Quantiles' 690 425 200 22 13 $C.soft $true 3 $true | Out-Null
Add-Text $s 'P50  7.85 Wh       P90  41.13 Wh       P95  109.36 Wh       P99  297.15 Wh' 100 457 800 22 12 $C.white $false 2 $false | Out-Null
Add-Text $s 'Largest miss | Actual 580 Wh  ·  Predicted 59.72 Wh  ·  Absolute error 520.28 Wh' 88 484 810 21 10 $C.amber $false 1 $false | Out-Null

# 13 — Anomaly detection
$s = New-Slide 'Anomaly Detection: Synthetic Experiment' 13 'Anomaly Detection'
Add-Text $s 'No ground-truth anomaly labels are available. We injected 60 positive spikes at 2 standard deviations into a copy of the test set.' 90 111 810 41 14 $C.soft $false 3 $true | Out-Null
$anomRows = @(
  @('Residual threshold: validation p99','0.026','0.017','0.020','0.263'),
  @('Isolation Forest (training p99)','0.035','0.083','0.049','0.022')
)
Add-Table $s @('Detector and threshold','Precision','Recall','F1','PR-AUC') $anomRows 66 178 828 110 @(365,112,112,112,127) 12 $true | Out-Null
Add-Text $s 'Detection Workflow' 730 327 165 23 16 $C.mint $true 3 $true | Out-Null
Add-Node $s "Forecast residuals`nValidation threshold" 615 366 250 55 $C.teal | Out-Null
Add-Node $s "Isolation Forest`nFit on training data" 320 366 250 55 $C.amber | Out-Null
Add-Node $s "Review signal`nNot an automated diagnosis" 67 366 215 55 $C.teal | Out-Null
Add-Line $s 615 394 574 394 $C.mint 1.2 $true | Out-Null; Add-Line $s 320 394 286 394 $C.mint 1.2 $true | Out-Null
Add-Text $s 'Scores are weak and depend on the synthetic injection design; they do not measure detection of natural events.' 103 455 790 30 14 $C.amber $true 3 $true | Out-Null

# 14 — Explainability & ablation
$s = New-Slide 'Interpretability and Feature Ablation' 14 'Interpretability'
Add-Text $s 'Feature ablation • validation on HistGradientBoosting' 450 109 445 24 13.5 $C.mint $true 3 $true | Out-Null
$ablRows = @(
  @('All features','63','22.121','57.325','0.596'),
  @('Without consumption history','49','22.265','57.473','0.594'),
  @('Top 20 mutual information','20','22.012','57.264','0.597')
)
Add-Table $s @('Feature set','Count','MAE','RMSE','R²') $ablRows 58 147 844 143 @(347,92,135,135,135) 11.5 $true | Out-Null
Add-Line $s 64 323 896 323 $C.border 0.8 | Out-Null
Add-Text $s 'Strongest Signal in the Selected Model' 533 345 360 25 16 $C.mint $true 3 $true | Out-Null
Add-Text $s 'Current Appliance Energy Use' 425 381 468 35 22 $C.white $true 3 $true | Out-Null
Add-Text $s 'Recent changes and lagged values follow. Reliance was measured as the change in mean absolute error on training data.' 110 431 783 44 13 $C.soft $false 3 $true | Out-Null
Add-Text $s 'Feature importance describes model reliance; it does not prove causation.' 370 479 520 21 12.5 $C.amber $true 3 $true | Out-Null

# 15 — GUI overview screenshot
$s = New-Slide 'Application Interface: Overview' 15 'Application'
$guiPath = Join-Path $assets 'gui_overview_wide.png'
if(-not (Test-Path $guiPath)){ $guiPath = Join-Path $assets 'gui_overview.png' }
Add-Image $s $guiPath 103 102 754 405 | Out-Null

# 16 — GUI model comparison screenshot
$s = New-Slide 'Application Interface: Model Comparison and Dark Theme' 16 'Application'
$guiDark = Join-Path $assets 'gui_models_dark_wide.png'
if(-not (Test-Path $guiDark)){ $guiDark = Join-Path $assets 'gui_models_dark.png' }
Add-Image $s $guiDark 103 102 754 405 | Out-Null

# 17 — Artifacts and verification
$s = New-Slide 'Artifacts and Reproducibility' 17 'Operation and Quality'
Add-Text $s 'Key Project Artifacts' 661 116 236 24 16 $C.mint $true 3 $true | Out-Null
$artRows = @(
  @('Model and inference','artifacts/wattwise_bundle.joblib'),
  @('Candidate comparison','reports/model_comparison.csv'),
  @('Metrics and diagnostics','reports/metrics/final_metrics.json'),
  @('Artifact integrity','reports/metrics/artifact_manifest.json'),
  @('Test errors','reports/metrics/largest_test_errors.csv')
)
Add-Table $s @('Purpose','File') $artRows 60 155 840 193 @(245,595) 11.3 $true | Out-Null
Add-Line $s 65 379 895 379 $C.border 0.8 | Out-Null
Add-Text $s 'Project Verification' 700 396 190 22 16 $C.mint $true 3 $true | Out-Null
Add-Text $s "28 automated tests passed • code-quality and build checks passed`nAll four notebooks executed • application pages and model comparison reviewed" 285 431 605 48 13.5 $C.white $false 3 $true | Out-Null
Add-Text $s 'Train: python -m wattwise.train  |  UI: streamlit run app.py' 100 485 650 20 11 $C.soft $false 1 $false | Out-Null

# 18 — Limits / close
$s = New-Slide 'Limitations and Next Steps' 18 'Conclusion'
$limitRows = @(
  @('Limited generalization','One home and a limited time span; performance in other homes is untested.'),
  @('Consumption peaks','Errors are larger during high-demand intervals and the later time block.'),
  @('Anomaly labels','No verified real-world events are available to validate waste detection.'),
  @('Next steps','More homes, verified labels, drift monitoring, and uncertainty evaluation.')
)
Add-Table $s @('Area','Finding') $limitRows 58 133 844 226 @(240,604) 13 $true | Out-Null
Add-Text $s 'Conclusion' 733 390 160 24 16 $C.mint $true 3 $true | Out-Null
Add-Text $s 'We forecast upcoming energy use and surface residuals to guide human review.' 105 423 790 43 20 $C.white $true 3 $true | Out-Null
Add-Text $s 'WattWise AI  •  Advanced Machine Learning' 260 478 630 23 12 $C.teal $true 2 $false | Out-Null

$pres.SaveAs($pptxPath,24)
$pres.Export($pngDir,'PNG',1280,720)
$pres.Close(); $ppt.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($pres) | Out-Null
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
Write-Output "Created $pptxPath"
Write-Output "Rendered slides to $pngDir"
