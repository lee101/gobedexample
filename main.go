// main.go - Complete GPU-accelerated embedding and search example
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/lee101/gobed/gpu"
)

// IndexTextsParallel processes texts in parallel with optimal GPU utilization
func IndexTextsParallel(pipeline *gpu.Pipeline, texts []string, chunkSize int) error {
	if len(texts) == 0 {
		return nil
	}

	log.Printf("🚀 Starting parallel GPU indexing of %d texts", len(texts))
	log.Printf("📦 Chunk size: %d (optimized for GPU)", chunkSize)
	
	start := time.Now()

	// Create chunks for optimal GPU batching
	chunks := make([][]string, 0)
	for i := 0; i < len(texts); i += chunkSize {
		end := i + chunkSize
		if end > len(texts) {
			end = len(texts)
		}
		chunks = append(chunks, texts[i:end])
	}

	log.Printf("📊 Created %d chunks (avg: %d texts/chunk)", len(chunks), len(texts)/len(chunks))

	// Parallel processing with controlled concurrency
	const maxConcurrent = 8 // Adjust based on GPU memory
	semaphore := make(chan struct{}, maxConcurrent)
	
	var wg sync.WaitGroup
	errors := make(chan error, len(chunks))
	progress := make(chan int, len(chunks))

	// Progress monitoring goroutine
	go func() {
		completed := 0
		for range progress {
			completed++
			if completed%5 == 0 || completed == len(chunks) {
				percent := float64(completed) / float64(len(chunks)) * 100
				elapsed := time.Since(start)
				rate := float64(completed*chunkSize) / elapsed.Seconds()
				
				log.Printf("📈 Progress: %.1f%% (%d/%d chunks, %.0f texts/sec)", 
					percent, completed, len(chunks), rate)
			}
		}
	}()

	// Process chunks in parallel
	for i, chunk := range chunks {
		wg.Add(1)
		go func(chunkNum int, chunkTexts []string) {
			defer wg.Done()

			// Acquire semaphore (limit concurrent GPU operations)
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Index the chunk
			chunkStart := time.Now()
			if err := pipeline.IndexTexts(chunkTexts); err != nil {
				errors <- fmt.Errorf("chunk %d failed: %w", chunkNum, err)
				return
			}

			chunkTime := time.Since(chunkStart)
			chunkRate := float64(len(chunkTexts)) / chunkTime.Seconds()
			
			// Report chunk completion (optional detailed logging)
			if len(chunks) <= 10 { // Only log details for smaller jobs
				log.Printf("✅ Chunk %d: %d texts in %v (%.0f texts/sec)", 
					chunkNum+1, len(chunkTexts), chunkTime, chunkRate)
			}

			progress <- 1
		}(i, chunk)
	}

	// Wait for all chunks to complete
	wg.Wait()
	close(errors)
	close(progress)

	// Check for errors
	var firstError error
	errorCount := 0
	for err := range errors {
		if firstError == nil {
			firstError = err
		}
		errorCount++
	}

	if firstError != nil {
		return fmt.Errorf("indexing failed (%d/%d chunks failed): %w", errorCount, len(chunks), firstError)
	}

	// Success metrics
	totalTime := time.Since(start)
	totalThroughput := float64(len(texts)) / totalTime.Seconds()

	log.Printf("✅ Parallel indexing complete!")
	log.Printf("   Total texts: %d", len(texts))
	log.Printf("   Total time: %v", totalTime)
	log.Printf("   Throughput: %.0f texts/sec", totalThroughput)
	log.Printf("   Chunks: %d", len(chunks))
	log.Printf("   Concurrency: %d", maxConcurrent)

	// Performance analysis
	if totalThroughput > 3000 {
		log.Printf("🚀 Excellent performance! GPU well utilized.")
	} else if totalThroughput > 1500 {
		log.Printf("✅ Good performance. Consider larger batches for even better GPU utilization.")
	} else {
		log.Printf("⚠️  Performance below expectations. Check GPU utilization and batch sizes.")
	}

	return nil
}

func main() {
	// Command line flags
	var (
		gpuServer   = flag.String("gpu-server", "http://localhost:5000", "GPU server URL")
		modelPath   = flag.String("model", "../gobed/model", "Path to model directory")
		dataFile    = flag.String("data", "large_data.txt", "Text file to index")
		interactive = flag.Bool("interactive", false, "Interactive search mode")
		benchmark   = flag.Bool("benchmark", false, "Run performance benchmark")
		startServer = flag.Bool("start-server", false, "Start GPU server automatically")
		batchSize   = flag.Int("batch-size", 256, "Batch size for GPU processing")
		maxTexts    = flag.Int("max-texts", 0, "Maximum number of texts to index (0 = all)")
		performanceTest = flag.Bool("performance-test", false, "Run with larger dataset for performance testing")
	)
	flag.Parse()

	// Start GPU server if requested
	if *startServer {
		if err := startGPUServer(); err != nil {
			log.Printf("Warning: Could not start GPU server: %v", err)
			log.Println("Please start it manually with: python3 ../gobed/gpu_search/gpu_search_server.py")
		} else {
			log.Println("✅ GPU server started")
			time.Sleep(3 * time.Second) // Wait for server to initialize
		}
	}

	// Initialize GPU pipeline with optimized settings
	optimizedBatchSize := 4096 // Much larger for GPU efficiency
	if *batchSize > 256 {
		optimizedBatchSize = *batchSize // Use custom if larger than default
	}
	
	config := gpu.Config{
		ModelPath:      *modelPath,
		GPUServerURL:   *gpuServer,
		BatchSize:      optimizedBatchSize, // Optimized batch size
		UseGPUIndexing: true,
		PreloadGPU:     true,  // Pre-allocate GPU memory
		MaxVectors:     1000000,
		GPUOnlyMode:    true, // Enable GPU-only mode for memory efficiency
	}

	pipeline, err := gpu.NewPipeline(config)
	if err != nil {
		log.Fatalf("Failed to create GPU pipeline: %v", err)
	}

	log.Printf("✅ Optimized GPU Pipeline initialized")
	log.Printf("   Batch size: %d (optimized for GPU)", optimizedBatchSize)
	log.Printf("   GPU-only mode: %t", config.GPUOnlyMode)
	log.Printf("   Preload GPU: %t", config.PreloadGPU)

	// Load or create sample data
	texts, err := loadTexts(*dataFile)
	if err != nil {
		log.Printf("Could not load data file %s: %v", *dataFile, err)
		log.Println("Creating sample data...")
		
		if *performanceTest {
			log.Println("🔥 Performance test mode: generating 10,000 texts...")
			texts = createLargeDataset(10000)
		} else {
			texts = createSampleData()
		}
		saveTexts(*dataFile, texts)
	}

	log.Printf("📚 Loaded %d texts for indexing", len(texts))
	
	// Limit texts if requested
	if *maxTexts > 0 && len(texts) > *maxTexts {
		texts = texts[:*maxTexts]
		log.Printf("📝 Limited to %d texts for testing", len(texts))
	}

	// Index texts on GPU using optimized parallel processing
	log.Println("🚀 Starting optimized GPU indexing...")
	
	// Determine optimal chunk size based on total texts and batch size
	chunkSize := optimizedBatchSize * 2 // Process 2 batches per chunk for efficiency
	if len(texts) < chunkSize {
		chunkSize = len(texts)
	}
	
	log.Printf("📊 Optimization settings:")
	log.Printf("   GPU batch size: %d", optimizedBatchSize)
	log.Printf("   Parallel chunk size: %d", chunkSize)
	log.Printf("   Max concurrent workers: 8")
	
	overallStart := time.Now()
	
	if err := IndexTextsParallel(pipeline, texts, chunkSize); err != nil {
		log.Fatalf("Failed to index texts: %v", err)
	}
	
	totalIndexTime := time.Since(overallStart)
	totalThroughput := float64(len(texts)) / totalIndexTime.Seconds()
	
	log.Printf("🎯 FINAL PERFORMANCE RESULTS:")
	log.Printf("   Total texts: %d", len(texts))
	log.Printf("   Total time: %v", totalIndexTime)
	log.Printf("   Final throughput: %.0f texts/sec", totalThroughput)
	
	// Compare to previous performance
	previousThroughput := 700.0 // Your current performance
	improvement := totalThroughput / previousThroughput
	log.Printf("   Improvement: %.1fx faster than baseline", improvement)
	
	if improvement > 5.0 {
		log.Printf("🚀 EXCELLENT! GPU optimization successful!")
	} else if improvement > 2.0 {
		log.Printf("✅ Good improvement. GPU utilization increased.")
	} else {
		log.Printf("⚠️  Limited improvement. Check GPU server and batch sizes.")
	}

	// Get pipeline statistics
	stats, err := pipeline.GetStats()
	if err == nil {
		log.Printf("📊 Pipeline Stats:")
		log.Printf("   Texts: %d", stats.NumTexts)
		log.Printf("   Embeddings: %d", stats.NumEmbeddings)
		log.Printf("   GPU Device: %s", stats.GPUDevice)
		log.Printf("   GPU Memory: %.1f MB", stats.GPUMemoryMB)
		if stats.CPUMemoryMB > 0 {
			log.Printf("   CPU Memory: %.1f MB", stats.CPUMemoryMB)
		} else {
			log.Printf("   CPU Memory: 0 MB (GPU-only mode)")
		}
		log.Printf("   Single Query: %.2fms (%.0f QPS)", stats.SingleQueryMS, stats.SingleQueryQPS)
		log.Printf("   Batch QPS: %.0f", stats.BatchQPS)
	}

	// Run benchmark if requested
	if *benchmark {
		runBenchmark(pipeline)
	}

	// Interactive or demo mode
	if *interactive {
		runInteractive(pipeline)
	} else {
		runDemo(pipeline)
	}
}

func startGPUServer() error {
	cmd := exec.Command("python3", "../gobed/gpu_search/gpu_search_server.py")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Start()
}

func loadTexts(filename string) ([]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var texts []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text != "" {
			texts = append(texts, text)
		}
	}

	return texts, scanner.Err()
}

func saveTexts(filename string, texts []string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for _, text := range texts {
		fmt.Fprintln(file, text)
	}
	return nil
}

func createLargeDataset(numTexts int) []string {
	baseTexts := []string{
		"Artificial intelligence is transforming technology and society",
		"Machine learning algorithms can process vast amounts of data",
		"Deep neural networks enable complex pattern recognition",
		"Natural language processing helps computers understand text",
		"Computer vision systems can identify objects in images",
		"Robotics combines AI with mechanical engineering",
		"Cloud computing provides scalable infrastructure",
		"Big data analytics reveals insights from large datasets",
		"Cybersecurity protects digital systems from threats",
		"Blockchain technology enables decentralized systems",
		"Internet of Things connects devices worldwide",
		"Quantum computing promises exponential speedups",
		"Renewable energy sources reduce carbon emissions",
		"Climate change affects global weather patterns",
		"Sustainable development balances economy and environment",
		"Healthcare technology improves patient outcomes",
		"Medical research develops new treatments",
		"Biotechnology advances genetic engineering",
		"Space exploration expands human knowledge",
		"Educational technology enhances learning",
	}
	
	texts := make([]string, numTexts)
	for i := 0; i < numTexts; i++ {
		baseText := baseTexts[i%len(baseTexts)]
		texts[i] = fmt.Sprintf("%s - document %d with additional content for testing", baseText, i+1)
	}
	
	return texts
}

func createSampleData() []string {
	return []string{
		// Technology
		"Artificial intelligence is transforming how we interact with technology",
		"Machine learning models can now understand and generate human language",
		"Deep learning has revolutionized computer vision and image recognition",
		"Neural networks are inspired by the structure of the human brain",
		"GPUs accelerate machine learning computations by orders of magnitude",
		"Transformer models have become the foundation of modern NLP",
		"BERT and GPT models have set new benchmarks in language understanding",
		"Computer vision can now detect objects with superhuman accuracy",
		"Reinforcement learning enables AI to master complex games and tasks",
		"Edge computing brings AI inference closer to data sources",
		
		// Science
		"Quantum computing promises exponential speedups for certain problems",
		"CRISPR gene editing technology allows precise DNA modifications",
		"Climate change is one of the most pressing challenges of our time",
		"Renewable energy sources are becoming increasingly cost-effective",
		"The human genome project mapped all human genes",
		"Stem cell research offers potential treatments for many diseases",
		"Dark matter makes up most of the universe's mass",
		"Black holes are regions where gravity is so strong nothing can escape",
		"The theory of evolution explains the diversity of life on Earth",
		"Photosynthesis converts sunlight into chemical energy in plants",
		
		// Health & Medicine
		"Regular exercise improves both physical and mental health",
		"A balanced diet is essential for maintaining good health",
		"Sleep is crucial for memory consolidation and healing",
		"Vaccines have saved millions of lives throughout history",
		"Antibiotics revolutionized medicine in the 20th century",
		"Mental health is as important as physical health",
		"Meditation can reduce stress and improve focus",
		"The immune system protects us from pathogens",
		"Personalized medicine tailors treatments to individual genetics",
		"Telemedicine makes healthcare more accessible to remote areas",
		
		// Business & Economics
		"Supply and demand determine prices in free markets",
		"Compound interest is one of the most powerful forces in finance",
		"Diversification reduces investment risk",
		"Startups drive innovation in the economy",
		"Digital transformation is reshaping traditional industries",
		"E-commerce has fundamentally changed retail",
		"Cryptocurrency introduces new forms of digital money",
		"Blockchain technology enables decentralized systems",
		"Remote work has become mainstream after the pandemic",
		"Automation is changing the nature of work",
		
		// Education
		"Lifelong learning is essential in the modern economy",
		"Online education makes knowledge accessible globally",
		"Critical thinking skills are more important than memorization",
		"STEM education prepares students for future careers",
		"Reading comprehension is fundamental to learning",
		"Project-based learning engages students more effectively",
		"Educational technology enhances traditional teaching methods",
		"Collaborative learning develops teamwork skills",
		"Personalized learning adapts to individual student needs",
		"Early childhood education has long-lasting impacts",
	}
}

func runDemo(pipeline *gpu.Pipeline) {
	fmt.Println("\n🔍 Running search demos...")
	
	queries := []string{
		"artificial intelligence and machine learning",
		"health benefits of exercise",
		"quantum physics and computing",
		"financial investment strategies",
		"climate and environment",
	}
	
	for _, query := range queries {
		fmt.Printf("\n📝 Query: %q\n", query)
		
		start := time.Now()
		results, err := pipeline.Search(query, 5)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}
		searchTime := time.Since(start)
		
		fmt.Printf("⏱️  Search time: %v\n", searchTime)
		fmt.Println("📊 Top 5 results:")
		
		for i, result := range results {
			fmt.Printf("   %d. [%.2f] %s\n", i+1, result.Score, result.Text)
		}
	}
	
	// Batch search demo
	fmt.Println("\n🚀 Running batch search...")
	start := time.Now()
	_, err := pipeline.BatchSearch(queries, 3)
	if err != nil {
		log.Printf("Batch search failed: %v", err)
		return
	}
	batchTime := time.Since(start)
	
	fmt.Printf("⏱️  Batch search time for %d queries: %v\n", len(queries), batchTime)
	fmt.Printf("💨 Average per query: %v\n", batchTime/time.Duration(len(queries)))
}

func runInteractive(pipeline *gpu.Pipeline) {
	fmt.Println("\n🔍 Interactive Search Mode")
	fmt.Println("Type your search query (or 'quit' to exit):")
	
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		
		query := strings.TrimSpace(scanner.Text())
		if query == "" {
			continue
		}
		if query == "quit" || query == "exit" {
			break
		}
		
		start := time.Now()
		results, err := pipeline.Search(query, 10)
		if err != nil {
			log.Printf("Search failed: %v", err)
			continue
		}
		searchTime := time.Since(start)
		
		fmt.Printf("\n⏱️  Search completed in %v\n", searchTime)
		fmt.Println("📊 Top 10 results:")
		
		for i, result := range results {
			fmt.Printf("   %2d. [%.2f] %s\n", i+1, result.Score, result.Text)
		}
	}
	
	fmt.Println("\n👋 Goodbye!")
}

func runBenchmark(pipeline *gpu.Pipeline) {
	fmt.Println("\n⚡ Running Performance Benchmark...")
	
	// Single query benchmark
	query := "machine learning and artificial intelligence"
	iterations := 100
	
	// Warmup
	for i := 0; i < 10; i++ {
		pipeline.Search(query, 10)
	}
	
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_, err := pipeline.Search(query, 10)
		if err != nil {
			log.Printf("Benchmark error: %v", err)
		}
	}
	elapsed := time.Since(start)
	
	avgLatency := elapsed / time.Duration(iterations)
	qps := float64(iterations) / elapsed.Seconds()
	
	fmt.Printf("\n📊 Single Query Performance:\n")
	fmt.Printf("   Iterations: %d\n", iterations)
	fmt.Printf("   Total time: %v\n", elapsed)
	fmt.Printf("   Average latency: %v\n", avgLatency)
	fmt.Printf("   Throughput: %.0f QPS\n", qps)
	
	// Batch benchmark
	batchSize := 32
	queries := make([]string, batchSize)
	for i := range queries {
		queries[i] = fmt.Sprintf("test query number %d", i)
	}
	
	batchIterations := 10
	start = time.Now()
	for i := 0; i < batchIterations; i++ {
		_, err := pipeline.BatchSearch(queries, 10)
		if err != nil {
			log.Printf("Batch benchmark error: %v", err)
		}
	}
	elapsed = time.Since(start)
	
	totalQueries := batchSize * batchIterations
	batchQPS := float64(totalQueries) / elapsed.Seconds()
	
	fmt.Printf("\n📊 Batch Query Performance (batch=%d):\n", batchSize)
	fmt.Printf("   Total queries: %d\n", totalQueries)
	fmt.Printf("   Total time: %v\n", elapsed)
	fmt.Printf("   Throughput: %.0f QPS\n", batchQPS)
	fmt.Printf("   Speedup vs single: %.1fx\n", batchQPS/qps)
}