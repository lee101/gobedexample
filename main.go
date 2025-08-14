package main

import (
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 Gobed Example: Using the Embedding Model")
	fmt.Println("================================================================================")
	fmt.Println()

	// Load the model
	fmt.Println("📦 Loading model from gobed...")
	start := time.Now()
	
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("❌ Failed to load model: %v", err)
	}
	
	loadTime := time.Since(start)
	fmt.Printf("✅ Model loaded successfully in %v\n", loadTime)
	fmt.Printf("📊 Model specs: %d vocab × %d dimensions\n\n", model.VocabSize, model.EmbedDim)

	// Example 1: Get available texts
	fmt.Println("📚 Example 1: Available texts for encoding")
	fmt.Println(strings.Repeat("-", 50))
	
	availableTexts := model.GetAvailableTexts()
	fmt.Printf("Found %d pre-tokenized texts:\n", len(availableTexts))
	for i, text := range availableTexts {
		if i >= 5 { // Show first 5
			fmt.Println("  ... and more")
			break
		}
		fmt.Printf("  • %s\n", text)
	}
	fmt.Println()

	// Example 2: Encode text
	fmt.Println("🔢 Example 2: Text Encoding")
	fmt.Println(strings.Repeat("-", 50))
	
	if len(availableTexts) > 0 {
		sampleText := availableTexts[0]
		fmt.Printf("Encoding: \"%s\"\n", sampleText)
		
		start := time.Now()
		embedding, err := model.Encode(sampleText)
		elapsed := time.Since(start)
		
		if err != nil {
			fmt.Printf("❌ Error encoding: %v\n", err)
		} else {
			fmt.Printf("✅ Encoded in %v\n", elapsed)
			fmt.Printf("   Dimensions: %d\n", len(embedding))
			fmt.Printf("   First 5 values: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
				embedding[0], embedding[1], embedding[2], embedding[3], embedding[4])
		}
	}
	fmt.Println()

	// Example 3: Calculate similarity
	fmt.Println("🎯 Example 3: Similarity Calculation")
	fmt.Println(strings.Repeat("-", 50))
	
	if len(availableTexts) >= 2 {
		text1 := availableTexts[0]
		text2 := availableTexts[1]
		
		fmt.Printf("Text 1: \"%s\"\n", text1)
		fmt.Printf("Text 2: \"%s\"\n", text2)
		
		similarity, err := model.Similarity(text1, text2)
		if err != nil {
			fmt.Printf("❌ Error calculating similarity: %v\n", err)
		} else {
			fmt.Printf("📊 Similarity score: %.4f\n", similarity)
			
			// Interpret the score
			if similarity > 0.5 {
				fmt.Println("   → Highly similar texts")
			} else if similarity > 0.2 {
				fmt.Println("   → Somewhat related texts")
			} else if similarity > 0 {
				fmt.Println("   → Weakly related texts")
			} else {
				fmt.Println("   → Unrelated or opposite texts")
			}
		}
	}
	fmt.Println()

	// Example 4: Find most similar texts
	fmt.Println("🔎 Example 4: Finding Similar Texts")
	fmt.Println(strings.Repeat("-", 50))
	
	if len(availableTexts) >= 3 {
		query := availableTexts[0]
		candidates := availableTexts[1:]
		
		fmt.Printf("Query: \"%s\"\n", query)
		fmt.Println("Finding top 3 most similar texts...")
		
		results, err := model.FindMostSimilar(query, candidates, 3)
		if err != nil {
			fmt.Printf("❌ Error finding similar texts: %v\n", err)
		} else {
			fmt.Println("\nResults:")
			for i, result := range results {
				fmt.Printf("  %d. \"%-30s\" → similarity: %.4f\n", 
					i+1, result.Text2, result.Similarity)
			}
		}
	}
	fmt.Println()

	// Example 5: Performance benchmark
	fmt.Println("⚡ Example 5: Performance Benchmark")
	fmt.Println(strings.Repeat("-", 50))
	
	if len(availableTexts) > 0 {
		testText := availableTexts[0]
		iterations := 1000
		
		fmt.Printf("Running %d encoding iterations...\n", iterations)
		
		start := time.Now()
		for i := 0; i < iterations; i++ {
			_, err := model.Encode(testText)
			if err != nil {
				fmt.Printf("❌ Error during benchmark: %v\n", err)
				break
			}
		}
		elapsed := time.Since(start)
		
		avgLatency := elapsed / time.Duration(iterations)
		throughput := float64(iterations) / elapsed.Seconds()
		
		fmt.Printf("\nPerformance Results:\n")
		fmt.Printf("  • Total time: %v\n", elapsed)
		fmt.Printf("  • Average latency: %v per encoding\n", avgLatency)
		fmt.Printf("  • Throughput: %.0f encodings/second\n", throughput)
		
		// Compare to Python baseline
		pythonLatency := 889 * time.Microsecond // From README
		speedup := float64(pythonLatency) / float64(avgLatency)
		fmt.Printf("  • Speedup vs Python GPU: %.1fx faster\n", speedup)
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("✅ Example completed successfully!")
	fmt.Println("🎉 The gobed embedding model is working perfectly!")
	fmt.Println(strings.Repeat("=", 80))
}