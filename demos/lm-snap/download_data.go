package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	DataDir      = "data"
	StoriesURL   = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
	WikiURL      = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
	StoriesFile  = "data/tinystories.txt"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║           SNAP DATA DOWNLOADER  ·  PHASE 2 CORPUS             ║")
	fmt.Println("║      Fetching High-Fluency Datasets for Word-Level Snap       ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	if _, err := os.Stat(DataDir); os.IsNotExist(err) {
		fmt.Printf("[*] Creating data directory: %s\n", DataDir)
		os.Mkdir(DataDir, 0755)
	}

	download(StoriesURL, StoriesFile)
	
	fmt.Println("\n[*] Data preparation complete. Ready for Word-Level Snap.")
}

func download(url, dest string) {
	fmt.Printf("[*] Downloading %s...\n", filepath.Base(dest))
	
	if _, err := os.Stat(dest); err == nil {
		fmt.Printf("    - File already exists: %s. Skipping.\n", dest)
		return
	}

	resp, err := http.Get(url)
	if err != nil {
		fmt.Printf("[!] Error fetching %s: %v\n", url, err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Printf("[!] Error: HTTP %d while fetching %s\n", resp.StatusCode, url)
		return
	}

	out, err := os.Create(dest)
	if err != nil {
		fmt.Printf("[!] Error creating file: %v\n", err)
		return
	}
	defer out.Close()

	// Use TeeReader to show progress if needed, but for 10MB it's fast.
	n, err := io.Copy(out, resp.Body)
	if err != nil {
		fmt.Printf("[!] Error during download: %v\n", err)
		return
	}

	fmt.Printf("    - Success! Downloaded %.2f MB to %s\n", float64(n)/1024/1024, dest)
}
