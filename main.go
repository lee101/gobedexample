package main

import (
	"fmt"
	"log"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("🚀 Gobed Example")
	fmt.Println("================")
	
	// Load the embedding model
	fmt.Print("Loading model... ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✓ (%.2fs)\n\n", time.Since(start).Seconds())

	// Example texts
	texts := []string{
		"Machine learning is fascinating.",
		"Deep learning models are powerful.",
		"The weather is nice today.",
		"Python is a programming language.",
		"Natural language processing",
	}

	// 1. Encode texts
	fmt.Println("📝 Encoding texts:")
	embeddings := make([][]float32, len(texts))
	for i, text := range texts {
		emb, err := model.Encode(text)
		if err != nil {
			fmt.Printf("   ❌ %s: %v\n", text, err)
			continue
		}
		embeddings[i] = emb
		fmt.Printf("   ✓ %s → [%.2f, %.2f, ...]\n", text, emb[0], emb[1])
	}
	fmt.Println()

	// 2. Calculate similarities
	fmt.Println("🎯 Similarity scores:")
	text1, text2 := texts[0], texts[1]
	similarity, err := model.Similarity(text1, text2)
	if err == nil {
		fmt.Printf("   '%s'\n   '%s'\n   → Similarity: %.4f\n", text1, text2, similarity)
	}
	fmt.Println()

	// 3. Find most similar
	fmt.Println("🔍 Most similar to '" + texts[0] + "':")
	results, err := model.FindMostSimilar(texts[0], texts[1:], 3)
	if err == nil {
		for i, r := range results {
			fmt.Printf("   %d. %s (%.4f)\n", i+1, r.Text2, r.Similarity)
		}
	}
	fmt.Println()

	// 4. Performance benchmark
	fmt.Println("⚡ Performance:")
	iterations := 10000
	testText := texts[0]
	
	start = time.Now()
	for i := 0; i < iterations; i++ {
		_, _ = model.Encode(testText)
	}
	elapsed := time.Since(start)
	
	avgLatency := elapsed / time.Duration(iterations)
	throughput := float64(iterations) / elapsed.Seconds()
	
	fmt.Printf("   • Latency: %v per encoding\n", avgLatency)
	fmt.Printf("   • Throughput: %.0f encodings/sec\n", throughput)
	fmt.Println()
	
	fmt.Println("✅ Done!")
}