package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"strings"
	"time"

	"github.com/lee101/gobed"
)

// GPUAccelerationDemo demonstrates GPU indexing concepts with gobed
func GPUAccelerationDemo() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 GPU-ACCELERATED INDEXING CONCEPTS DEMONSTRATION")
	fmt.Println("================================================================================")
	fmt.Printf("System: %d CPUs\n", runtime.NumCPU())
	fmt.Println()

	// Load the embedding model
	fmt.Print("Loading model... ")
	start := time.Now()
	model, err := gobed.LoadModel()
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("✓ (%.2fs)\n\n", time.Since(start).Seconds())

	// Get available texts for testing
	availableTexts := model.GetAvailableTexts()
	if len(availableTexts) < 10 {
		fmt.Println("Not enough reference texts available")
		return
	}

	// Use first 10 texts for demonstration
	testTexts := availableTexts[:10]

	// 1. Demonstrate batch vs single processing
	demonstrateBatchProcessing(model, testTexts)

	// 2. Demonstrate INT8 quantization benefits
	demonstrateINT8Benefits()

	// 3. Demonstrate similarity search performance
	demonstrateSimilaritySearch(model, testTexts)

	// 4. Show GPU performance projections
	showGPUProjections(len(testTexts))

	fmt.Println("\n✅ Demonstration complete!")
}

func demonstrateBatchProcessing(model *gobed.EmbeddingModel, texts []string) {
	fmt.Printf("%s\n", strings.Repeat("=", 80))
	fmt.Printf("⚡ BATCH PROCESSING DEMONSTRATION\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Single processing (simulating CPU)
	fmt.Printf("\n1️⃣ Single Processing (CPU-like):\n")
	singleStart := time.Now()
	
	for i := 0; i < len(texts)-1; i++ {
		_, _ = model.Similarity(texts[i], texts[i+1])
	}
	
	singleTime := time.Since(singleStart)
	singleThroughput := float64(len(texts)-1) / singleTime.Seconds()
	
	fmt.Printf("   Time: %.2fms\n", float64(singleTime.Nanoseconds())/1e6)
	fmt.Printf("   Throughput: %.0f comparisons/sec\n", singleThroughput)

	// Simulated batch processing (GPU-like)
	fmt.Printf("\n2️⃣ Batch Processing (GPU simulation):\n")
	
	// GPU would be 10-25x faster for batch operations
	gpuSpeedup := 15.0
	gpuTime := time.Duration(float64(singleTime) / gpuSpeedup)
	gpuThroughput := singleThroughput * gpuSpeedup
	
	fmt.Printf("   Estimated Time: %.2fms\n", float64(gpuTime.Nanoseconds())/1e6)
	fmt.Printf("   Estimated Throughput: %.0f comparisons/sec\n", gpuThroughput)
	fmt.Printf("\n📈 Estimated GPU Speedup: %.1fx faster\n", gpuSpeedup)
}

func demonstrateINT8Benefits() {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("📊 INT8 QUANTIZATION BENEFITS\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Simulate embeddings
	numVectors := 100000
	dim := 384
	
	// Calculate memory usage
	fp32Memory := numVectors * dim * 4
	fp16Memory := numVectors * dim * 2
	int8Memory := numVectors * dim * 1
	
	fmt.Printf("\n💾 Memory Usage for %d vectors (dimension %d):\n", numVectors, dim)
	fmt.Printf("   FP32: %.1f MB (baseline)\n", float64(fp32Memory)/(1024*1024))
	fmt.Printf("   FP16: %.1f MB (%.1fx reduction)\n", 
		float64(fp16Memory)/(1024*1024), float64(fp32Memory)/float64(fp16Memory))
	fmt.Printf("   INT8: %.1f MB (%.1fx reduction)\n", 
		float64(int8Memory)/(1024*1024), float64(fp32Memory)/float64(int8Memory))

	// Simulate quantization process
	fmt.Printf("\n🔢 Quantization Process:\n")
	
	// Generate sample values
	sampleValues := make([]float32, 100)
	for i := range sampleValues {
		sampleValues[i] = rand.Float32()*2 - 1 // Range [-1, 1]
	}
	
	// Find min/max
	minVal := float32(math.MaxFloat32)
	maxVal := float32(-math.MaxFloat32)
	for _, val := range sampleValues {
		if val < minVal {
			minVal = val
		}
		if val > maxVal {
			maxVal = val
		}
	}
	
	// Calculate quantization parameters
	scale := (maxVal - minVal) / 255.0
	zeroPoint := int8(-math.Round(float64(minVal / scale)))
	
	fmt.Printf("   Scale factor: %.6f\n", scale)
	fmt.Printf("   Zero point: %d\n", zeroPoint)
	
	// Calculate average error
	totalError := float32(0)
	for _, val := range sampleValues {
		quantized := int8(math.Round(float64(val/scale)) + float64(zeroPoint))
		dequantized := float32(quantized-zeroPoint) * scale
		error := math.Abs(float64(val - dequantized))
		totalError += float32(error)
	}
	avgError := totalError / float32(len(sampleValues))
	
	fmt.Printf("   Average error: %.6f\n", avgError)
	fmt.Printf("   Accuracy retained: %.1f%%\n", (1.0-avgError)*100)

	fmt.Printf("\n⚡ Performance Benefits:\n")
	fmt.Printf("   • 4x less memory bandwidth required\n")
	fmt.Printf("   • 2-4x faster operations on GPUs with INT8 support\n")
	fmt.Printf("   • Enables larger models on same hardware\n")
}

func demonstrateSimilaritySearch(model *gobed.EmbeddingModel, texts []string) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("🔍 SIMILARITY SEARCH PERFORMANCE\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	if len(texts) < 2 {
		fmt.Println("Not enough texts for search demonstration")
		return
	}

	queryText := texts[0]
	fmt.Printf("\nQuery: \"%s\"\n", queryText)
	fmt.Printf("\nSearching through %d texts...\n", len(texts)-1)

	// Perform similarity calculations
	searchStart := time.Now()
	results := make([]struct {
		text       string
		similarity float32
	}, 0)

	for i := 1; i < len(texts); i++ {
		similarity, err := model.Similarity(queryText, texts[i])
		if err == nil {
			results = append(results, struct {
				text       string
				similarity float32
			}{texts[i], similarity})
		}
	}

	searchTime := time.Since(searchStart)

	// Sort results by similarity (simple bubble sort for small dataset)
	for i := 0; i < len(results)-1; i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].similarity > results[i].similarity {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Display top results
	fmt.Printf("\nTop 3 most similar texts:\n")
	for i := 0; i < 3 && i < len(results); i++ {
		fmt.Printf("  %d. %.4f - %s\n", i+1, results[i].similarity, results[i].text)
	}

	fmt.Printf("\n⏱️  Performance:\n")
	fmt.Printf("   CPU Search Time: %.2fms\n", float64(searchTime.Nanoseconds())/1e6)
	fmt.Printf("   CPU Throughput: %.0f searches/sec\n", 1000.0/float64(searchTime.Nanoseconds()/1e6))
	
	// GPU projections
	fmt.Printf("\n🚀 GPU Performance (estimated):\n")
	fmt.Printf("   GPU FP32: ~%.3fms (20x speedup)\n", float64(searchTime.Nanoseconds())/1e6/20)
	fmt.Printf("   GPU INT8: ~%.3fms (50x speedup)\n", float64(searchTime.Nanoseconds())/1e6/50)
	fmt.Printf("   GPU Batch-100: ~%.3fms per query (500x speedup)\n", float64(searchTime.Nanoseconds())/1e6/500)
}

func showGPUProjections(dataSize int) {
	fmt.Printf("\n%s\n", strings.Repeat("=", 80))
	fmt.Printf("📈 GPU PERFORMANCE PROJECTIONS\n")
	fmt.Printf("%s\n", strings.Repeat("=", 80))

	// Different dataset sizes
	sizes := []int{1000, 10000, 100000, 1000000}
	
	fmt.Printf("\n%-15s | %12s | %12s | %12s | %10s\n",
		"Dataset Size", "CPU Time", "GPU FP32", "GPU INT8", "Speedup")
	fmt.Printf("%s\n", strings.Repeat("-", 75))

	for _, size := range sizes {
		// Estimate times based on linear scaling
		cpuTime := float64(size) * 0.001 // 1ms per 1000 vectors
		gpuFP32Time := cpuTime / 20      // 20x speedup
		gpuINT8Time := cpuTime / 50      // 50x speedup
		
		fmt.Printf("%-15d | %11.1fms | %11.3fms | %11.3fms | %9.0fx\n",
			size, cpuTime, gpuFP32Time, gpuINT8Time, cpuTime/gpuINT8Time)
	}

	fmt.Printf("\n🎯 Key Insights:\n")
	fmt.Printf("   • GPU efficiency increases with dataset size\n")
	fmt.Printf("   • INT8 provides both memory and compute benefits\n")
	fmt.Printf("   • Batch processing maximizes GPU utilization\n")
	fmt.Printf("   • Real-time search possible even with millions of vectors\n")

	fmt.Printf("\n💡 Hardware Recommendations:\n")
	fmt.Printf("   • Small (<10M vectors): NVIDIA T4 (best value)\n")
	fmt.Printf("   • Medium (10-30M): RTX 3090/4090 (excellent price/performance)\n")
	fmt.Printf("   • Large (>30M): A100 (maximum performance)\n")

	fmt.Printf("\n⚙️  Optimization Tips:\n")
	fmt.Printf("   • Use batch size of 5000 for optimal GPU utilization\n")
	fmt.Printf("   • Enable INT8 quantization for production\n")
	fmt.Printf("   • Keep data on GPU to avoid transfer overhead\n")
	fmt.Printf("   • Use IVF indexing for datasets >1M vectors\n")
}

func main() {
	rand.Seed(42)
	GPUAccelerationDemo()
}