@echo off
set OPENCLAW_PUPPET_CONFIG=%~dp0puppet.json
python -m autonomous.run %*
