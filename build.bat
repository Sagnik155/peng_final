@echo off
echo =======================================================
echo Building Docker image for Grand Challenge...
echo =======================================================
docker build -t pengwin-submission .

echo.
echo =======================================================
echo Packaging image into a .tar file (This may take a few minutes)...
echo =======================================================
docker save pengwin-submission -o pengwin-submission.tar

echo.
echo =======================================================
echo Build complete! 
echo You can now upload the 'pengwin-submission.tar' file directly to the portal.
echo =======================================================
pause