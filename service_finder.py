import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

def parse_query(query, model, tokenizer):
    """Parse query using Mistral 7B (Query Parsing AI Agent)."""
    prompt = f"""Extract the service type, location, and price range (min and max) from this query: {query}
Output in JSON format with these fields:
- service: The type of service needed
- location: The city and state
- min_price: Minimum price (default 0.0)
- max_price: Maximum price (default 1000.0)"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.1,
        do_sample=True
    )
    parsed = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from model output
    try:
        # Find JSON-like structure in the output
        start_idx = parsed.find("{")
        end_idx = parsed.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = parsed[start_idx:end_idx]
            result = json.loads(json_str)
            return {
                "service": result.get("service", "Unknown"),
                "location": result.get("location", "Unknown"),
                "min_price": float(result.get("min_price", 0.0)),
                "max_price": float(result.get("max_price", 1000.0))
            }
    except:
        pass
    
    # Fallback to rule-based parsing if AI parsing fails
    return parse_query_rule_based(query)

def parse_query_rule_based(query):
    """Fallback rule-based query parsing."""
    query = query.lower()
    # Detect service
    service = "Unknown"
    if "plumber" in query:
        service = "Plumber"
    elif "real estate" in query or "listing agent" in query:
        service = "Real Estate Agent"
    elif "tutor" in query:
        service = "Tutor"
    elif "carpenter" in query:
        service = "Carpenter"
    elif "painter" in query:
        service = "Painter"
    elif "electrician" in query:
        service = "Electrician"
    
    # Detect location
    location = "Unknown"
    locations = {
        "san francisco": "San Francisco, CA",
        "sf": "San Francisco, CA",
        "sfo": "San Francisco, CA",
        "mountain view": "Mountain View, CA",
        "mv": "Mountain View, CA",
        "palo alto": "Palo Alto, CA",
        "pa": "Palo Alto, CA",
        "santa clara": "Santa Clara, CA",
        "sc": "Santa Clara, CA",
        "san jose": "San Jose, CA",
        "sj": "San Jose, CA",
        "saratoga": "Saratoga, CA",
        "los gatos": "Los Gatos, CA",
        "lg": "Los Gatos, CA",
        "atherton": "Atherton, CA",
        "berkeley": "Berkeley, CA"
    }
    
    for loc_key, loc_value in locations.items():
        if loc_key in query:
            location = loc_value
            break
    
    # Detect price range
    min_price = 0.0
    max_price = 1000.0
    words = query.split()
    for i, word in enumerate(words):
        if word == "between" and i + 1 < len(words) and i + 3 < len(words):
            try:
                min_price = float(words[i + 1].replace("$", ""))
                max_price = float(words[i + 3].replace("$", ""))
            except (ValueError, IndexError):
                pass
        elif word.isdigit():
            max_price = float(word)
    
    return {"service": service, "location": location, "min_price": min_price, "max_price": max_price}

def load_providers(file_path):
    """Load providers from JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {file_path}.")
        return []

def load_external_providers(file_path, embeddings):
    """Load external providers and create FAISS index (Data Retrieval AI Agent)."""
    try:
        with open(file_path, "r") as file:
            providers = json.load(file)
        texts = [f"{p['service']} in {p['location']}: {p['description']}" for p in providers]
        vector_store = FAISS.from_texts(texts, embeddings)
        return providers, vector_store
    except Exception as e:
        print(f"Error loading external providers: {e}")
        return [], None

def retrieve_external_providers(query, providers, vector_store, embeddings, k=3):
    """Retrieve relevant external providers using RAG."""
    if not vector_store:
        return []
    try:
        docs = vector_store.similarity_search(query, k=k)
        retrieved = []
        for doc in docs:
            # Extract provider info from document
            text = doc.page_content
            for provider in providers:
                if (provider["service"] in text and 
                    provider["location"] in text and 
                    provider["description"] in text):
                    retrieved.append(provider)
                    break
        return retrieved
    except Exception as e:
        print(f"Error retrieving external providers: {e}")
        return []

def match_providers(parsed_query, providers, external_providers, model):
    """Match providers using similarity-based scoring (Matching AI Agent)."""
    try:
        # Prepare query text
        query_text = f"{parsed_query['service']} in {parsed_query['location']}"
        query_embedding = model.encode(query_text)
        
        # Combine local and external providers
        all_providers = providers + external_providers
        
        # Filter by location and price range first
        filtered_providers = [
            p for p in all_providers
            if p["location"] == parsed_query["location"]
            and parsed_query["min_price"] <= p["price"] <= parsed_query["max_price"]
        ]
        
        if not filtered_providers:
            return []
        
        # Compute similarity scores
        provider_texts = [f"{p['service']}: {p['description']}" for p in filtered_providers]
        provider_embeddings = model.encode(provider_texts)
        
        similarities = torch.cosine_similarity(
            torch.tensor(query_embedding).unsqueeze(0),
            torch.tensor(provider_embeddings)
        )
        
        # Sort by similarity
        scored_providers = list(zip(filtered_providers, similarities))
        scored_providers.sort(key=lambda x: x[1], reverse=True)
        
        return [p for p, _ in scored_providers[:3]]
    except Exception as e:
        print(f"Error in matching providers: {e}")
        return []

def save_matches(matches, output_path):
    """Save matches to JSON file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            json.dump(matches, file, indent=4)
    except Exception as e:
        print(f"Error saving matches: {e}")

def save_feedback(query, top_match, helpful, output_path):
    """Save user feedback to JSON file."""
    feedback_entry = {
        "query": query,
        "top_match": top_match,
        "helpful": helpful
    }
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            with open(output_path, "r") as file:
                feedback = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            feedback = []
        
        feedback.append(feedback_entry)
        
        with open(output_path, "w") as file:
            json.dump(feedback, file, indent=4)
    except Exception as e:
        print(f"Error saving feedback: {e}")

def main():
    """Run the Bay Area Service Finder pipeline with AI agents."""
    print("Initializing AI agents...")
    
    # Initialize Mistral 7B (Query Parsing AI Agent)
    print("Loading Mistral 7B model...")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    # Initialize sentence-transformers (Matching AI Agent)
    print("Loading sentence-transformers model...")
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize embeddings for RAG (Data Retrieval AI Agent)
    print("Initializing embeddings for RAG...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load providers
    print("Loading provider data...")
    providers = load_providers("data/providers.json")
    external_providers, vector_store = load_external_providers("data/external_providers.json", embeddings)
    
    # Test queries
    test_queries = [
        "I need a painter in Mountain View between $40 and $90",
        "Find an electrician in Berkeley under $100",
        "I need a real estate agent in Palo Alto for luxury homes"
    ]
    
    for query in test_queries:
        print("\n" + "="*50)
        print(f"Testing query: {query}")
        print("="*50)
        
        # Parse query with Mistral 7B
        parsed = parse_query(query, model, tokenizer)
        print(f"Parsed: {parsed}")
        
        # Retrieve external providers with RAG
        retrieved_providers = retrieve_external_providers(query, external_providers, vector_store, embeddings)
        
        # Match providers
        matches = match_providers(parsed, providers, retrieved_providers, sentence_model)
        if not matches:
            print("No matches found.")
            continue
        
        # Display matches
        print("\nTop Matches:")
        for i, match in enumerate(matches, 1):
            print(f"\nMatch {i}:")
            print(f"Service: {match['service']}")
            print(f"Location: {match['location']}")
            print(f"Price: ${match['price']}")
            print(f"Description: {match['description']}")
        
        # Save matches
        save_matches(matches, "results/matches.json")
        print(f"\nResults saved to results/matches.json")
        
        # Collect feedback
        if matches:
            while True:
                helpful = input("\nWas the top match helpful? (yes/no): ").strip().lower()
                if helpful in ["yes", "no"]:
                    save_feedback(query, matches[0], helpful, "results/feedback.json")
                    print("Feedback saved to results/feedback.json")
                    break
                else:
                    print("Please enter 'yes' or 'no'")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main() 