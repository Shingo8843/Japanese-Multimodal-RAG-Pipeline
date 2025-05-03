import os
from typing import List, Dict, Union
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from chromadb.config import Settings
from PIL import Image
import config
import numpy as np
import base64
from io import BytesIO

class JapaneseRAGPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self._load_models()
        self._setup_chromadb()
        self._initialize_demo_data()

    def _load_models(self):
        """Load all required models"""
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=self.device
        )
        
        print("Loading reranker model...")
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL,
            device=self.device
        )
        
        print("Loading Qwen-VL model...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.LLM_MODEL,
            trust_remote_code=True
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL,
            device_map="auto",
            trust_remote_code=True
        )

    def _setup_chromadb(self):
        """Initialize ChromaDB client and collection"""
        os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=config.CHROMA_PERSIST_DIR,
            settings=Settings(allow_reset=True)
        )
        
        # Get or create collection
        try:
            self.collection = self.chroma_client.get_collection(
                name=config.CHROMA_COLLECTION_NAME
            )
            print(f"Found existing collection: {config.CHROMA_COLLECTION_NAME}")
        except Exception as e:  # Catch all exceptions
            print(f"Collection not found, creating new one: {str(e)}")
            self.collection = self.chroma_client.create_collection(
                name=config.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {config.CHROMA_COLLECTION_NAME}")

    def _initialize_demo_data(self):
        """Initialize collection with some demo data if empty"""
        if self.collection.count() == 0:
            print("Initializing demo data...")
            demo_texts = [
                "こんにちは"
            ]
            
            # Generate embeddings
            embeddings = [
                self.embed_text(text) for text in demo_texts
            ]
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=demo_texts,
                ids=[f"demo_{i}" for i in range(len(demo_texts))]
            )
            print(f"Added {len(demo_texts)} demo documents to collection")

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        return self.embedding_model.encode(text).tolist()

    def rerank_documents(self, query: str, documents: List[str]) -> List[Dict]:
        """Rerank documents based on relevance to query"""
        if not documents:
            return []
            
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Combine documents with their scores
        ranked_docs = list(zip(documents, scores))
        # Sort by score in descending order
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [{"text": doc, "score": float(score)} for doc, score in ranked_docs]

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        """Prepare image for Qwen-VL model"""
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Convert to RGB if needed
        if image_np.shape[-1] == 4:  # RGBA
            image_np = image_np[..., :3]  # Remove alpha channel
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor

    def generate_response(self, prompt: str, image: Image.Image = None) -> str:
        """Generate response using Qwen-VL"""
        try:
            if image:
                # Prepare image tensor
                image_tensor = self._prepare_image(image)
                
                # Tokenize with image
                inputs = self.tokenizer(
                    prompt,
                    images=image_tensor,
                    return_tensors="pt"
                ).to(self.device)
            else:
                # Tokenize without image
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt"
                ).to(self.device)
                
            with torch.inference_mode():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Clean up the response by removing the input prompt
            response = response.replace(prompt, "").strip()
            return response
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return f"申し訳ありません。エラーが発生しました。Error: {str(e)}"

    def add_document(self, text: str, image_data: str = None, metadata: dict = None) -> None:
        """Add a document to the collection with optional image data"""
        try:
            # Generate embedding for the text
            text_embedding = np.array(self.embed_text(text))
            
            # If image data is provided, generate image embedding
            if image_data:
                # Decode base64 image data
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
                
                # Prepare image for Qwen-VL
                image_tensor = self._prepare_image(image)
                
                # Generate image embedding using Qwen-VL
                with torch.no_grad():
                    # Use the model's image processing
                    inputs = self.tokenizer(
                        text,
                        images=image_tensor,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get the image features from the model's forward pass
                    outputs = self.llm(**inputs, output_hidden_states=True)
                    image_features = outputs.hidden_states[-1][:, :inputs.input_ids.shape[1]].mean(dim=1)
                    # Convert bfloat16 to float32
                    image_features = image_features.to(torch.float32)
                    image_embedding = image_features[0].cpu().numpy()
                
                # Resize image embedding to match text embedding size
                if len(image_embedding) > len(text_embedding):
                    # Take average of chunks to reduce dimension
                    chunk_size = len(image_embedding) // len(text_embedding)
                    image_embedding = np.mean(image_embedding.reshape(-1, chunk_size), axis=1)[:len(text_embedding)]
                else:
                    # Pad with zeros if needed
                    padding = np.zeros(len(text_embedding) - len(image_embedding))
                    image_embedding = np.concatenate([image_embedding, padding])
                
                # Normalize both embeddings
                text_embedding = text_embedding / np.linalg.norm(text_embedding)
                image_embedding = image_embedding / np.linalg.norm(image_embedding)
                
                # Combine text and image embeddings with equal weight
                combined_embedding = ((text_embedding + image_embedding) / 2).tolist()
            else:
                combined_embedding = text_embedding.tolist()
            
            # Prepare document data
            doc_data = {
                "embeddings": [combined_embedding],
                "documents": [text],
                "ids": [f"doc_{self.collection.count()}"]
            }
            
            # Add metadata if provided
            if metadata:
                doc_data["metadatas"] = [metadata]
            
            # Add to collection
            self.collection.add(**doc_data)
            print(f"Added document: {text[:30]}...")
            
        except Exception as e:
            print(f"Error adding document: {str(e)}")
            raise

    def process_query(self, query: str, image: Image.Image = None) -> str:
        """Process a query through the entire RAG pipeline"""
        try:
            # Generate query embedding
            query_embedding = np.array(self.embed_text(query))
            
            # If image is provided, generate image embedding and combine
            if image:
                image_tensor = self._prepare_image(image)
                with torch.no_grad():
                    # Use the model's image processing
                    inputs = self.tokenizer(
                        query,
                        images=image_tensor,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get the image features from the model's forward pass
                    outputs = self.llm(**inputs, output_hidden_states=True)
                    image_features = outputs.hidden_states[-1][:, :inputs.input_ids.shape[1]].mean(dim=1)
                    # Convert bfloat16 to float32
                    image_features = image_features.to(torch.float32)
                    image_embedding = image_features[0].cpu().numpy()
                
                # Resize image embedding to match text embedding size
                if len(image_embedding) > len(query_embedding):
                    # Take average of chunks to reduce dimension
                    chunk_size = len(image_embedding) // len(query_embedding)
                    image_embedding = np.mean(image_embedding.reshape(-1, chunk_size), axis=1)[:len(query_embedding)]
                else:
                    # Pad with zeros if needed
                    padding = np.zeros(len(query_embedding) - len(image_embedding))
                    image_embedding = np.concatenate([image_embedding, padding])
                
                # Normalize both embeddings
                query_embedding = query_embedding / np.linalg.norm(query_embedding)
                image_embedding = image_embedding / np.linalg.norm(image_embedding)
                
                # Combine query and image embeddings with equal weight
                query_embedding = ((query_embedding + image_embedding) / 2).tolist()
            else:
                query_embedding = query_embedding.tolist()
            
            # Retrieve relevant documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(config.TOP_K_RETRIEVAL, self.collection.count())
            )
            
            if not results["documents"]:
                return "申し訳ありません。関連する情報が見つかりませんでした。"
            
            # Rerank documents
            reranked_docs = self.rerank_documents(query, results["documents"][0])
            
            # Take top N documents after reranking
            top_docs = reranked_docs[:config.TOP_N_RERANK]
            
            # Prepare context for LLM
            context = "\n".join([f"Document {i+1}: {doc['text']}" for i, doc in enumerate(top_docs)])
            prompt = f"以下の文脈に基づいて、質問に日本語で答えてください。\n\n文脈:\n{context}\n\n質問: {query}\n\n回答:"
            
            # Generate response
            response = self.generate_response(prompt, image)
            return response
            
        except Exception as e:
            print(f"Error in process_query: {str(e)}")
            return f"申し訳ありません。エラーが発生しました。Error: {str(e)}" 