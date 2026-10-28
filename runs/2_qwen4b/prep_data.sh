#!/bin/bash
# Данные для 4B (запуск 22.10.2026): русские и английские веб-тексты, без диалогов. Отсюда и потеря навыка ассистента.
cd ~/ternary/release
~/jev/.venv/bin/python prepare_data.py --model Qwen/Qwen3.5-4B --out ~/q4b/data --ru 16e6 --en 6e6 --val 0.3e6
