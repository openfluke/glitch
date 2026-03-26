@echo off
echo Running exhaustive layer tests...
(echo 3 & echo 0 & echo 1) | go run glitch.go > output.txt 2>&1
echo Tests complete. Results saved to output.txt
type output.txt
