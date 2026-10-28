@echo off
chcp 65001 >nul
cd /d "%~dp0"
set OUT=results_%COMPUTERNAME%.txt
echo Замер тернарных моделей на процессоре. Займёт около 20 минут, компьютер в это время лучше не нагружать.
echo Итог будет в файле %OUT%
echo. > %OUT%
powershell -NoProfile -Command "'Процессор: ' + (Get-CimInstance Win32_Processor).Name; 'Память, МГц: ' + ((Get-CimInstance Win32_PhysicalMemory | Select-Object -ExpandProperty Speed) -join ', ')" >> %OUT%

rem 1. качество: перплексия с переупаковкой и без (должна совпасть с точностью до 0,5%%)
for %%m in (tern08-TQ2_0 bonsai2b-TQ2_0) do (
  echo == %%m, перплексия, переупаковка вкл >> %OUT%
  bin\llama-perplexity.exe -m models\%%m.gguf -f val_ru.txt -c 512 --chunks 8 -t 4 2>&1 | findstr /C:"Final estimate" >> %OUT%
  echo == %%m, перплексия, переупаковка выкл >> %OUT%
  bin\llama-perplexity.exe -m models\%%m.gguf -f val_ru.txt -c 512 --chunks 8 -t 4 -nr 2>&1 | findstr /C:"Final estimate" >> %OUT%
)

rem 2. скорость: pp512 = чтение запроса, tg128 = генерация (токенов в секунду)
for %%m in (tern08-TQ2_0 bonsai2b-TQ2_0 bonsai2b-Q4_0 qwen2b-Q3_K_M tern4b-TQ2_0) do (
  for %%t in (4 8) do (
    for %%r in (1 0) do (
      echo == %%m, потоков %%t, переупаковка %%r
      echo == %%m, потоков %%t, переупаковка %%r >> %OUT%
      bin\llama-bench.exe -m models\%%m.gguf -t %%t -p 512 -n 128 -r 2 --repack %%r 2>nul | findstr /C:"pp512" /C:"tg128" >> %OUT%
    )
  )
)
echo.
echo Готово. Пришлите файл %OUT%
pause
