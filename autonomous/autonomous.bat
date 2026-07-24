@echo off
set OPENCLAW_AUTONOMOUS_CONFIG=%~dp0autonomous.json
python -m autonomous.run %*
