import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os

class ServiceMatchmaker:
    def __init__(self):
        """Initialize the ServiceMatchmaker with required models and data."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        self.load_providers()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def load_model(self):
        """Load the quantized Mistral 7B model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            self.model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mistral-7B-v0.1",
                device_map="auto",
                load_in_4bit=True,
                torch_dtype=torch.float16
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def load_providers(self):
        """Load provider data from JSON file."""
        try:
            with open("data/providers.json", "r") as f:
                self.providers = json.load(f)
        except FileNotFoundError:
            print("Error: providers.json not found in data directory")
            raise
        except json.JSONDecodeError:
            print("Error: Invalid JSON format in providers.json")
            raise

    def parse_query(self, query):
        """Parse natural language query using Mistral 7B."""
        prompt = f"Extract the service type, location, and budget from this query: {query}\nOutput as JSON:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.1,
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            # Find the last occurrence of "Output as JSON:" and extract everything after it
            json_str = response.split("Output as JSON:")[-1].strip()
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            print("Error parsing model output as JSON")
            return None

    def generate_embeddings(self, texts):
        """Generate embeddings for a list of texts using sentence-transformers."""
        return self.embedding_model.encode(texts, convert_to_tensor=True)

    def match_providers(self, parsed_query):
        """Match providers based on parsed query using embeddings and constraints."""
        if not parsed_query:
            return []

        # Filter providers by location and price
        filtered_providers = [
            p for p in self.providers
            if p["location"] == parsed_query["location"] and
            p["price"] <= parsed_query["price"] and
            p["service"] == parsed_query["service"]
        ]

        if not filtered_providers:
            return []

        # Generate embeddings for query and provider descriptions
        query_embedding = self.generate_embeddings([parsed_query["service"] + " " + parsed_query["location"]])[0]
        provider_embeddings = self.generate_embeddings([p["description"] for p in filtered_providers])

        # Calculate cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0),
            provider_embeddings
        )

        # Get top 3 matches
        top_indices = similarities.argsort(descending=True)[:3]
        matches = []
        
        for idx in top_indices:
            provider = filtered_providers[idx]
            matches.append({
                "provider": provider,
                "similarity_score": float(similarities[idx])
            })

        return matches

    def save_results(self, query, parsed_query, matches):
        """Save matching results to JSON file."""
        results = {
            "query": query,
            "parsed_query": parsed_query,
            "matches": matches
        }
        
        os.makedirs("results", exist_ok=True)
        with open("results/matches.json", "w") as f:
            json.dump(results, f, indent=2)

def main():
    """Main function to run the service matchmaker."""
    try:
        matchmaker = ServiceMatchmaker()
        
        # Example query
        query = "I need a plumber in Austin under $100"
        print(f"Query: {query}")
        
        # Parse query
        parsed_query = matchmaker.parse_query(query)
        print(f"Parsed: {json.dumps(parsed_query, indent=2)}")
        
        # Match providers
        matches = matchmaker.match_providers(parsed_query)
        
        # Print results
        for i, match in enumerate(matches, 1):
            provider = match["provider"]
            print(f"\nMatch {i}: {provider['service']} in {provider['location']} (${provider['price']})")
            print(f"Description: {provider['description']}")
            print(f"Similarity Score: {match['similarity_score']:.2f}")
        
        # Save results
        matchmaker.save_results(query, parsed_query, matches)
        
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()

# Next Steps for Iteration 2:
# 1. Scrape 1,000 provider profiles using BeautifulSoup or Yelp API.
# 2. Implement RAG with LangChain to fetch real-time data from X or Yelp.
# 3. Optimize inference with torch.cuda.amp and quantize to 8-bit with Hugging Face Optimum.
# 4. Write a C++ CUDA kernel for cosine similarity to enhance GPU performance.
# 5. Share fine-tuned model and dataset on Hugging Face for open-source contribution. 