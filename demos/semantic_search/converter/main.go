package main

import (
	"archive/zip"
	"bufio"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/openfluke/loom/poly"
)

const (
	GloveURL      = "http://nlp.stanford.edu/data/glove.6B.zip"
	ZipFile       = "glove.6B.zip"
	VectorFile    = "glove.6B.300d.txt"
	SafetensorOut = "glove_300d.safetensors"
	VocabOut      = "vocab.txt"
)

func main() {
	if _, err := os.Stat(SafetensorOut); err == nil {
		fmt.Printf("✅ %s already exists, skipping conversion.\n", SafetensorOut)
		return
	}

	if _, err := os.Stat(ZipFile); os.IsNotExist(err) {
		fmt.Printf("⏳ Downloading GloVe 6B (Zip: ~822MB)... This could take a minute.\n")
		err := downloadFile(ZipFile, GloveURL)
		if err != nil {
			panic(fmt.Sprintf("Failed to download: %v", err))
		}
	}

	fmt.Printf("⏳ Extracting %s...\n", VectorFile)
	err := extractFile(ZipFile, VectorFile)
	if err != nil {
		panic(fmt.Sprintf("Failed to extract: %v", err))
	}

	fmt.Printf("⏳ Converting %s to Safetensors format...\n", VectorFile)
	err = convertToSafetensors(VectorFile, SafetensorOut, VocabOut)
	if err != nil {
		panic(fmt.Sprintf("Failed to convert: %v", err))
	}

	fmt.Printf("✅ Success! Model saved as %s and Vocab saved as %s\n", SafetensorOut, VocabOut)
}

func downloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func extractFile(zipPath, filename string) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return err
	}
	defer r.Close()

	for _, f := range r.File {
		if f.Name == filename {
			rc, err := f.Open()
			if err != nil {
				return err
			}
			defer rc.Close()

			out, err := os.Create(f.Name)
			if err != nil {
				return err
			}
			defer out.Close()

			_, err = io.Copy(out, rc)
			return err
		}
	}
	return fmt.Errorf("file %s not found in zip", filename)
}

func convertToSafetensors(txtPath, safePath, vocabPath string) error {
	file, err := os.Open(txtPath)
	if err != nil {
		return err
	}
	defer file.Close()

	vocabFile, err := os.Create(vocabPath)
	if err != nil {
		return err
	}
	defer vocabFile.Close()

	var values []float32
	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 1024*1024)

	wordCount := 0
	dim := 0

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		if len(parts) < 2 {
			continue
		}

		word := parts[0]
		vocabFile.WriteString(word + "\n")

		if dim == 0 {
			dim = len(parts) - 1
		}

		for i := 1; i < len(parts); i++ {
			f, _ := strconv.ParseFloat(parts[i], 32)
			values = append(values, float32(f))
		}
		wordCount++

		if wordCount % 50000 == 0 {
			fmt.Printf("...processed %d words\n", wordCount)
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	fmt.Printf("Creating safetensor with shape [%d, %d]\n", wordCount, dim)
	
	tensors := make(map[string]poly.TensorWithShape)
	tensors["embeddings.weight"] = poly.TensorWithShape{
		Values: values,
		Shape:  []int{wordCount, dim},
		DType:  "F32",
	}

	return poly.SaveSafetensors(safePath, tensors)
}
