// generate_large_dataset.go - Create 200K examples for GPU testing
package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	fmt.Println("🚀 Generating 200K examples for GPU benchmarking...")
	
	texts := generateLargeDataset(200000)
	
	fmt.Printf("📝 Generated %d texts\n", len(texts))
	
	// Save to file
	file, err := os.Create("large_data.txt")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	
	for _, text := range texts {
		fmt.Fprintln(file, text)
	}
	
	fmt.Printf("✅ Saved to large_data.txt (%.1f MB)\n", 
		float64(getTotalSize(texts))/1024/1024)
}

func getTotalSize(texts []string) int {
	total := 0
	for _, text := range texts {
		total += len(text)
	}
	return total
}

func generateLargeDataset(count int) []string {
	rand.Seed(time.Now().UnixNano())
	
	// Base templates for different categories
	templates := map[string][]string{
		"technology": {
			"Advanced %s technology enables %s applications in %s",
			"The %s framework provides %s capabilities for %s development",
			"Modern %s systems utilize %s algorithms for %s optimization",
			"Innovative %s solutions deliver %s performance in %s environments",
			"Next-generation %s platforms support %s workflows for %s industries",
			"Cutting-edge %s tools facilitate %s processes in %s domains",
			"Revolutionary %s methods improve %s efficiency in %s operations",
			"State-of-the-art %s architectures enable %s scaling for %s applications",
			"Emerging %s technologies transform %s practices in %s sectors",
			"Sophisticated %s implementations enhance %s capabilities for %s users",
		},
		"science": {
			"Research in %s reveals %s patterns affecting %s phenomena",
			"Scientific studies on %s demonstrate %s effects in %s systems",
			"Laboratory analysis of %s shows %s properties under %s conditions",
			"Experimental data from %s indicates %s relationships in %s processes",
			"Theoretical models of %s predict %s behaviors in %s environments",
			"Observational studies of %s confirm %s hypotheses about %s mechanisms",
			"Clinical trials using %s validate %s treatments for %s disorders",
			"Field research on %s documents %s variations in %s populations",
			"Comparative analysis of %s identifies %s factors in %s development",
			"Longitudinal studies of %s track %s changes over %s periods",
		},
		"business": {
			"Strategic %s initiatives drive %s growth in %s markets",
			"Operational %s improvements increase %s efficiency across %s divisions",
			"Financial %s strategies optimize %s returns for %s stakeholders",
			"Marketing %s campaigns enhance %s engagement with %s audiences",
			"Supply chain %s innovations reduce %s costs in %s operations",
			"Human resources %s programs develop %s skills for %s roles",
			"Customer service %s solutions improve %s satisfaction in %s channels",
			"Product development %s processes accelerate %s delivery to %s markets",
			"Risk management %s frameworks mitigate %s exposures in %s activities",
			"Corporate %s policies ensure %s compliance across %s jurisdictions",
		},
		"health": {
			"Medical %s treatments improve %s outcomes for %s patients",
			"Preventive %s measures reduce %s risks in %s populations",
			"Diagnostic %s procedures detect %s conditions through %s methods",
			"Therapeutic %s interventions address %s symptoms of %s diseases",
			"Rehabilitation %s programs restore %s function after %s injuries",
			"Nutritional %s guidelines promote %s health in %s demographics",
			"Exercise %s routines enhance %s fitness for %s individuals",
			"Mental health %s services support %s wellbeing through %s approaches",
			"Pharmaceutical %s research develops %s medications for %s conditions",
			"Healthcare %s systems deliver %s care to %s communities",
		},
		"education": {
			"Educational %s methods enhance %s learning in %s environments",
			"Teaching %s strategies improve %s comprehension of %s subjects",
			"Learning %s technologies support %s instruction for %s students",
			"Curriculum %s design promotes %s understanding of %s concepts",
			"Assessment %s tools measure %s progress in %s competencies",
			"Student %s services provide %s support for %s development",
			"Faculty %s training improves %s delivery of %s content",
			"Academic %s programs prepare %s professionals for %s careers",
			"Research %s opportunities advance %s knowledge in %s fields",
			"Institutional %s policies ensure %s quality in %s programs",
		},
	}
	
	// Word pools for filling templates
	wordPools := map[string][]string{
		"tech_words": {
			"artificial intelligence", "machine learning", "deep learning", "neural networks",
			"blockchain", "cloud computing", "edge computing", "quantum computing",
			"cybersecurity", "data science", "big data", "analytics",
			"automation", "robotics", "IoT", "5G", "AR/VR", "microservices",
			"kubernetes", "docker", "devops", "serverless", "API", "database",
		},
		"science_words": {
			"molecular biology", "genetics", "biochemistry", "neuroscience",
			"physics", "chemistry", "astronomy", "geology", "ecology",
			"immunology", "pharmacology", "pathology", "epidemiology",
			"statistics", "mathematics", "computational biology", "bioinformatics",
			"crystallography", "spectroscopy", "microscopy", "proteomics",
		},
		"business_words": {
			"strategy", "operations", "finance", "marketing", "sales",
			"management", "leadership", "innovation", "digital transformation",
			"supply chain", "logistics", "procurement", "manufacturing",
			"consulting", "analytics", "optimization", "efficiency",
			"revenue", "profit", "growth", "scalability", "sustainability",
		},
		"health_words": {
			"cardiology", "oncology", "neurology", "pediatrics", "surgery",
			"radiology", "pathology", "psychiatry", "rehabilitation",
			"nutrition", "fitness", "wellness", "prevention", "treatment",
			"diagnosis", "therapy", "medication", "clinical trials",
			"public health", "epidemiology", "genetics", "immunology",
		},
		"education_words": {
			"curriculum", "pedagogy", "assessment", "learning", "teaching",
			"research", "scholarship", "methodology", "technology",
			"online learning", "distance education", "blended learning",
			"competency", "skill development", "critical thinking",
			"collaboration", "communication", "creativity", "innovation",
			"STEM", "humanities", "social sciences", "arts",
		},
		"generic_words": {
			"advanced", "innovative", "efficient", "effective", "comprehensive",
			"integrated", "scalable", "robust", "flexible", "reliable",
			"sustainable", "optimized", "streamlined", "enhanced", "improved",
			"sophisticated", "cutting-edge", "state-of-the-art", "next-generation",
			"revolutionary", "transformative", "disruptive", "emerging",
		},
		"domains": {
			"healthcare", "finance", "education", "manufacturing", "retail",
			"transportation", "energy", "telecommunications", "agriculture",
			"government", "defense", "entertainment", "media", "gaming",
			"real estate", "insurance", "consulting", "hospitality",
			"aerospace", "automotive", "pharmaceuticals", "biotechnology",
		},
	}
	
	texts := make([]string, 0, count)
	categories := []string{"technology", "science", "business", "health", "education"}
	
	for i := 0; i < count; i++ {
		// Select random category and template
		category := categories[rand.Intn(len(categories))]
		template := templates[category][rand.Intn(len(templates[category]))]
		
		// Fill template with random words
		text := fillTemplate(template, wordPools)
		
		// Add some variation with numbers and additional context
		if rand.Float32() < 0.3 {
			number := rand.Intn(1000) + 1
			text = fmt.Sprintf("%s. Study %d shows %s", text, number, 
				getRandomPhrase(wordPools))
		}
		
		if rand.Float32() < 0.2 {
			year := rand.Intn(30) + 1995
			text = fmt.Sprintf("Since %d, %s", year, text)
		}
		
		texts = append(texts, text)
		
		// Progress indicator
		if i%10000 == 0 {
			fmt.Printf("Generated %d/%d texts...\n", i, count)
		}
	}
	
	return texts
}

func fillTemplate(template string, wordPools map[string][]string) string {
	// Count placeholders
	placeholders := strings.Count(template, "%s")
	
	// Get random words to fill placeholders
	words := make([]interface{}, placeholders)
	for i := 0; i < placeholders; i++ {
		poolName := getRandomPoolName(wordPools)
		pool := wordPools[poolName]
		words[i] = pool[rand.Intn(len(pool))]
	}
	
	return fmt.Sprintf(template, words...)
}

func getRandomPoolName(wordPools map[string][]string) string {
	poolNames := make([]string, 0, len(wordPools))
	for name := range wordPools {
		poolNames = append(poolNames, name)
	}
	return poolNames[rand.Intn(len(poolNames))]
}

func getRandomPhrase(wordPools map[string][]string) string {
	phrases := []string{
		"significant improvements in efficiency",
		"reduced costs and increased performance",
		"enhanced user experience and satisfaction",
		"improved accuracy and reliability",
		"faster processing and better results",
		"increased scalability and flexibility",
		"better integration and compatibility",
		"enhanced security and compliance",
		"improved productivity and innovation",
		"reduced complexity and maintenance",
	}
	return phrases[rand.Intn(len(phrases))]
}

// Additional function to generate numerical data patterns
func generateNumericalDataset(count int) []string {
	texts := make([]string, 0, count)
	
	for i := 0; i < count; i++ {
		// Generate various numerical patterns
		switch rand.Intn(10) {
		case 0:
			// Mathematical sequences
			seq := make([]string, rand.Intn(10)+5)
			start := rand.Intn(100)
			step := rand.Intn(10) + 1
			for j := range seq {
				seq[j] = strconv.Itoa(start + j*step)
			}
			texts = append(texts, fmt.Sprintf("Sequence: %s", strings.Join(seq, ", ")))
			
		case 1:
			// Statistical data
			mean := rand.Float64()*100 + 50
			stddev := rand.Float64()*20 + 5
			texts = append(texts, fmt.Sprintf("Dataset with mean %.2f and standard deviation %.2f", mean, stddev))
			
		case 2:
			// Time series data
			values := make([]string, rand.Intn(20)+10)
			for j := range values {
				values[j] = fmt.Sprintf("%.2f", rand.Float64()*100)
			}
			texts = append(texts, fmt.Sprintf("Time series data: %s", strings.Join(values, ", ")))
			
		default:
			// Regular text with numbers
			num1 := rand.Intn(1000)
			num2 := rand.Intn(1000)
			operation := []string{"plus", "minus", "times", "divided by"}[rand.Intn(4)]
			texts = append(texts, fmt.Sprintf("Calculate %d %s %d for optimization", num1, operation, num2))
		}
	}
	
	return texts
}