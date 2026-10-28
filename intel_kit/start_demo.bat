@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Демонстрация: откроется страница http://127.0.0.1:8090 (модель выбирается на вкладке «Попробовать самому»).
echo Чтобы остановить, закройте это окно.
start "" http://127.0.0.1:8090
bin\llama-server.exe --models-preset models.ini --models-max 1 --host 127.0.0.1 --port 8090 --path web
