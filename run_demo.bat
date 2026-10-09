@echo off
REM Convenience wrapper: Windows finds the wrong "python" first on PATH (msys64's, which lacks
REM cv2/ultralytics) -- this always uses the correct one. Run from anywhere, no path typing.
"D:\DOANMINHHIEU\STUDIES\Code\python.exe" "%~dp0src\demo_helmet.py" %*
