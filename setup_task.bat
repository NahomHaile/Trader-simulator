@echo off
schtasks /create /tn "StockPaperTrader" /tr "cmd /c \"\"C:\Users\nhail\AppData\Local\Programs\Python\Python311\python.exe\" \"C:\Users\nhail\OneDrive\Desktop\stock\paper_trade.py\" >> \"C:\Users\nhail\OneDrive\Desktop\stock\paper_trade_log.txt\" 2>&1\"" /sc WEEKLY /d MON,TUE,WED,THU,FRI /st 09:31 /f
echo.
echo Task created. Verify with:
echo schtasks /query /tn "StockPaperTrader" /fo LIST
pause
