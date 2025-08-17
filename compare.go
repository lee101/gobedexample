package main

import (
	"fmt"
	"log"

	"github.com/lee101/gobed"
)

func main() {
	// Minimal example: compare a few string pairs
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("failed to load model: %v", err)
	}

	pairs := [][2]string{
		{"Machine learning is fascinating.", "Deep learning models are powerful."},
		{"Python is a programming language.", "JavaScript runs in browsers."},
		{"Hello world", "Pizza tastes delicious."},
	}

	fmt.Println("Gobed: simple similarity examples")
	fmt.Println("=================================")
	for i, p := range pairs {
		sim, err := model.Similarity(p[0], p[1])
		if err != nil {
			fmt.Printf("%d) error comparing: %v\n", i+1, err)
			continue
		}
		distance := 1 - sim
		fmt.Printf("%d) \"%s\" \n   vs \"%s\" \n   → similarity: %.4f, distance: %.4f\n\n", i+1, p[0], p[1], sim, distance)
	}
}
