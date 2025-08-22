// gpu_stress.go - Stress test GPU to show sustained utilization
package main

import (
	"flag"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/lee101/gobed/gpu"
)

func main() {
	var (
		modelPath  = flag.String("model", "../gobed/model", "Path to model directory")
		concurrent = flag.Int("concurrent", 10, "Number of concurrent search threads")
		duration   = flag.Int("duration", 30, "Test duration in seconds")
	)
	flag.Parse()

	fmt.Println("🔥 GPU Stress Test - Sustained High Utilization")
	fmt.Printf("   Concurrent threads: %d\n", *concurrent)
	fmt.Printf("   Duration: %d seconds\n", *duration)
	fmt.Println("   Monitor with: nvidia-smi dmon -s um -d 1")
	fmt.Println()

	// Initialize pipeline with large dataset
	config := gpu.Config{
		ModelPath:      *modelPath,
		GPUServerURL:   "http://localhost:5000",
		BatchSize:      64,
		UseGPUIndexing: true,
		GPUOnlyMode:    true,
		MaxVectors:     200000,
	}

	pipeline, err := gpu.NewPipeline(config)
	if err != nil {
		log.Fatalf("Failed to create pipeline: %v", err)
	}

	// Load large dataset for stress testing
	fmt.Println("📚 Loading large dataset...")
	texts := generateStressTestData(50000)
	
	start := time.Now()
	err = pipeline.IndexTexts(texts)
	if err != nil {
		log.Fatalf("Failed to index texts: %v", err)
	}
	indexTime := time.Since(start)
	
	fmt.Printf("✅ Indexed %d texts in %v\n", len(texts), indexTime)

	// Prepare test queries
	queries := []string{
		"artificial intelligence and machine learning algorithms",
		"quantum computing and cryptography security",
		"sustainable energy and climate change solutions", 
		"financial markets and investment strategies",
		"healthcare technology and medical research",
		"space exploration and rocket technology",
		"robotics automation and manufacturing",
		"blockchain cryptocurrency and digital finance",
		"biotechnology genetic engineering research",
		"renewable energy solar wind power",
	}

	fmt.Printf("\n🚀 Starting stress test with %d concurrent threads...\n", *concurrent)
	fmt.Println("Watch GPU utilization with: nvidia-smi dmon")
	
	// Stress test metrics
	var (
		totalQueries int64
		totalLatency time.Duration
		mu           sync.Mutex
	)

	// Start worker goroutines
	var wg sync.WaitGroup
	stop := make(chan bool)

	for i := 0; i < *concurrent; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			queryCount := 0
			workerLatency := time.Duration(0)
			
			for {
				select {
				case <-stop:
					mu.Lock()
					totalQueries += int64(queryCount)
					totalLatency += workerLatency
					mu.Unlock()
					fmt.Printf("Worker %d: %d queries, avg %.2fms\n", 
						workerID, queryCount, float64(workerLatency.Nanoseconds())/float64(queryCount)/1e6)
					return
				default:
					// Perform search
					query := queries[queryCount%len(queries)]
					start := time.Now()
					
					_, err := pipeline.Search(query, 10)
					if err != nil {
						continue
					}
					
					latency := time.Since(start)
					workerLatency += latency
					queryCount++
					
					// Small delay to prevent overwhelming
					time.Sleep(10 * time.Millisecond)
				}
			}
		}(i)
	}

	// Also run batch searches for higher GPU utilization
	wg.Add(1)
	go func() {
		defer wg.Done()
		
		batchCount := 0
		for {
			select {
			case <-stop:
				fmt.Printf("Batch worker: %d batch searches completed\n", batchCount)
				return
			default:
				start := time.Now()
				_, err := pipeline.BatchSearch(queries, 10)
				if err != nil {
					continue
				}
				elapsed := time.Since(start)
				batchCount++
				
				fmt.Printf("Batch %d: %d queries in %v (%.0f QPS)\n", 
					batchCount, len(queries), elapsed, float64(len(queries)*1000)/float64(elapsed.Milliseconds()))
				
				time.Sleep(100 * time.Millisecond)
			}
		}
	}()

	// Run for specified duration
	time.Sleep(time.Duration(*duration) * time.Second)
	close(stop)
	wg.Wait()

	// Final stats
	avgLatency := totalLatency / time.Duration(totalQueries)
	qps := float64(totalQueries) / float64(*duration)
	
	fmt.Printf("\n📊 Stress Test Results:\n")
	fmt.Printf("   Duration: %d seconds\n", *duration)
	fmt.Printf("   Total queries: %d\n", totalQueries)
	fmt.Printf("   Average latency: %.2fms\n", float64(avgLatency.Nanoseconds())/1e6)
	fmt.Printf("   Throughput: %.0f QPS\n", qps)
	fmt.Printf("   Concurrent threads: %d\n", *concurrent)
}

func generateStressTestData(count int) []string {
	texts := make([]string, count)
	
	categories := []string{
		"artificial intelligence", "machine learning", "quantum computing",
		"blockchain technology", "renewable energy", "biotechnology",
		"space exploration", "robotics automation", "financial markets",
		"healthcare innovation", "climate change", "cybersecurity",
	}
	
	actions := []string{
		"research reveals", "studies demonstrate", "analysis shows",
		"experiments prove", "investigations find", "data indicates",
		"results suggest", "findings confirm", "evidence supports",
	}
	
	contexts := []string{
		"significant improvements in", "breakthrough advances in",
		"revolutionary changes to", "innovative solutions for",
		"enhanced capabilities in", "optimized performance of",
		"sustainable development of", "next-generation approaches to",
	}
	
	for i := 0; i < count; i++ {
		category := categories[i%len(categories)]
		action := actions[(i/len(categories))%len(actions)]
		context := contexts[(i/(len(categories)*len(actions)))%len(contexts)]
		
		texts[i] = fmt.Sprintf("Advanced %s %s %s modern applications. Research study %d demonstrates remarkable potential for scalable solutions.",
			category, action, context, i+1000)
	}
	
	return texts
}