// torch_test.go - Test LibTorch integration
package main

import (
	"fmt"
	"log"
	"os"
	"time"

	"github.com/lee101/gobed/gpu"
)

func testTorchPipeline() {
	fmt.Println("🧪 Testing LibTorch GPU Pipeline")

	// Check if model exists
	modelPath := "../gobed/model/simple_gpu_search_module.pt"
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		fmt.Printf("❌ Model not found: %s\n", modelPath)
		fmt.Println("Please run: cd ../gobed/gpu_search && python3 simple_search_module.py")
		return
	}

	// Configure pipeline
	config := gpu.Config{
		ModelPath:      "../gobed/model",
		BatchSize:      64,
		UseGPUIndexing: false, // Use CPU for embedding, GPU for search
		MaxVectors:     10000,
		GPUOnlyMode:    true,
	}

	// Create LibTorch pipeline
	pipeline, err := gpu.NewTorchPipeline(config)
	if err != nil {
		log.Fatalf("Failed to create torch pipeline: %v", err)
	}
	defer pipeline.Close()

	fmt.Println("✅ LibTorch pipeline initialized")

	// Test data
	testTexts := []string{
		"Machine learning is transforming technology",
		"Artificial intelligence powers modern applications",
		"Deep learning models understand human language",
		"Computer vision recognizes objects accurately",
		"Natural language processing enables text analysis",
		"Neural networks mimic brain functionality",
		"GPU acceleration speeds up computations",
		"CUDA kernels optimize parallel processing",
		"PyTorch provides flexible deep learning tools",
		"TensorFlow enables large-scale machine learning",
	}

	// Index texts
	fmt.Println("🚀 Indexing texts...")
	start := time.Now()
	
	if err := pipeline.IndexTexts(testTexts); err != nil {
		log.Fatalf("Failed to index texts: %v", err)
	}
	
	indexTime := time.Since(start)
	fmt.Printf("✅ Indexed %d texts in %v\n", len(testTexts), indexTime)

	// Get stats
	stats, err := pipeline.GetStats()
	if err != nil {
		log.Printf("Failed to get stats: %v", err)
	} else {
		fmt.Printf("📊 Pipeline Stats:\n")
		fmt.Printf("   Texts: %d\n", stats.NumTexts)
		fmt.Printf("   Embeddings: %d\n", stats.NumEmbeddings)
		fmt.Printf("   GPU Device: %s\n", stats.GPUDevice)
		fmt.Printf("   GPU Memory: %.1f MB\n", stats.GPUMemoryMB)
	}

	// Test single search
	fmt.Println("\n🔍 Testing single search...")
	query := "deep learning neural networks"
	
	start = time.Now()
	results, err := pipeline.Search(query, 3)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}
	searchTime := time.Since(start)

	fmt.Printf("Query: %q\n", query)
	fmt.Printf("Search time: %v\n", searchTime)
	fmt.Println("Results:")
	for i, result := range results {
		fmt.Printf("  %d. [%.3f] %s\n", i+1, result.Score, result.Text)
	}

	// Test batch search
	fmt.Println("\n🚀 Testing batch search...")
	queries := []string{
		"machine learning algorithms",
		"computer vision technology",
		"GPU parallel processing",
	}

	start = time.Now()
	batchResults, err := pipeline.BatchSearch(queries, 2)
	if err != nil {
		log.Fatalf("Batch search failed: %v", err)
	}
	batchTime := time.Since(start)

	fmt.Printf("Batch search time for %d queries: %v\n", len(queries), batchTime)
	fmt.Printf("Average per query: %v\n", batchTime/time.Duration(len(queries)))

	for i, query := range queries {
		fmt.Printf("\nQuery %d: %q\n", i+1, query)
		for j, result := range batchResults[i] {
			fmt.Printf("  %d. [%.3f] %s\n", j+1, result.Score, result.Text)
		}
	}

	fmt.Println("\n✅ LibTorch pipeline test completed successfully!")
}

func main() {
	testTorchPipeline()
}