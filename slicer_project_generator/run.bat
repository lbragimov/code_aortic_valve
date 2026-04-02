@echo off
cd /d "%~dp0..\"
.venv\Scripts\python.exe -m slicer_project_generator.main
pause