@echo on

REM === Set timestamp for log file ===
for /f %%A in ('powershell -NoProfile -Command "Get-Date -Format yyyy_MM_dd_HH_mm_ss"') do set TIMESTAMP=%%A

REM === Parse Command-Line Arguments ===

:parse_args
if /i "%~1"=="--COMMAND" (
    set "COMMAND="
    :collect_command_arg
    if "%~2"=="" goto done
    set "COMMAND=%COMMAND% %~2"
    shift
    goto collect_command_arg
)
shift


REM == GO TO BEGINING OF THE LOOP ==
goto parse_args

REM == LOOP OVER ==
:done

echo %COMMAND%


REM === Activate Conda environment ===
CALL "C:\Users\ricar\anaconda3\Scripts\activate.bat" "C:\Users\ricar\anaconda3\envs\rl"

REM Guarantee module is installed

echo Starting experiment at %TIME%

%COMMAND%

exit()

pause
