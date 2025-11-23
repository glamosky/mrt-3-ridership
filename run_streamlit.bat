@echo off
echo Starting Streamlit Dashboard...
echo.

REM Activate micromamba and the environment
call C:\Users\franj\micromamba\Scripts\activate.bat
call micromamba activate mrt

REM Change to the script directory
cd /d "%~dp0"

REM Run Streamlit
streamlit run metrotren.py --server.port 8501 --server.headless true

pause