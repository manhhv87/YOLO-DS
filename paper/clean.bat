@echo off
setlocal EnableExtensions

REM Di chuyển về thư mục chứa file .bat (thư mục project)
cd /d "%~dp0"

echo ========= Cleaning LaTeX aux files under:
echo   %cd%
echo.

REM Danh sách phần mở rộng cần xóa (không dùng tham số hàm cho đỡ rối)
for %%E in (
  aux log toc out synctex.gz bbl blg fls fdb_latexmk
  lof lot nav snm vrb xdv bcf run.xml thm idx ilg ind
  acn acr alg glg glo gls ist loa lol .bbl-SAVE-ERROR
) do (
  echo [EXT] *.%%E

  REM Xóa ở thư mục gốc
  del /Q /F ".\*.%%E" 2>nul

  REM Xóa đệ quy tất cả file có đuôi %%E trong cây thư mục
  for /R %%F in (*.%%E) do (
    del /Q /F "%%F" 2>nul
  )
)

echo.
echo Done.
endlocal
