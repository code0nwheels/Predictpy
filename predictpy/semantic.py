"""
Semantic memory module for learning user speaking patterns and predicting thought completion.
Uses ChromaDB for storing and retrieving semantic embeddings.
"""
import os
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import re

import chromadb
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize


class SemanticMemory:
    """
    Manages semantic storage and retrieval of user text patterns.
    Stores embeddings of user thoughts/sentences for intelligent completion.
    """
    
    def __init__(self, db_path: str = "~/.predictpy/chroma", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic memory system.
        
        Args:
            db_path: Path to ChromaDB storage directory
            model_name: Sentence transformer model to use for embeddings
        """
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(self.db_path, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name="user_thoughts",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize sentence transformer
        self.encoder = SentenceTransformer(model_name)
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def store_text(self, text: str, context_before: str = "", context_after: str = "",
                   text_type: str = "general", tags: Optional[List[str]] = None) -> int:
        """
        Store text and learn semantic patterns.
        
        Args:
            text: Text to store and learn from
            context_before: Optional context that came before this text
            context_after: Optional context that comes after this text
            text_type: Type of text (email, chat, document, etc.)
            tags: Optional tags for categorization
            
        Returns:
            Number of thoughts stored
        """
        if not text.strip():
            return 0
        
        # Split text into semantic units (thoughts)
        thoughts = self._split_thoughts(text)
        stored_count = 0
        
        for i, thought in enumerate(thoughts):
            # Skip very short thoughts
            if len(thought.split()) < 3:
                continue
            
            # Check if we should store this (avoid duplicates)
            if not self._should_store(thought):
                continue
            
            # Generate embedding
            embedding = self.encoder.encode(thought).tolist()
            
            # Prepare metadata
            metadata = {
                "type": self._classify_thought(thought),
                "text_type": text_type,
                "timestamp": datetime.now().isoformat(),
                "word_count": len(thought.split()),
                "context_before": thoughts[i-1] if i > 0 else context_before,
                "context_after": thoughts[i+1] if i < len(thoughts)-1 else context_after,
                "completion": self._extract_completion(thought),
                "tags": ",".join(tags or [])
            }
            
            # Generate unique ID
            doc_id = hashlib.md5(thought.encode('utf-8')).hexdigest()
            
            try:
                self.collection.add(
                    documents=[thought],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[doc_id]
                )
                stored_count += 1
            except Exception as e:
                logging.warning(f"Failed to store thought: {e}")
        
        return stored_count
    
    def predict_completion(self, partial_text: str, n_results: int = 5,
                          context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Predict completion of a thought/paragraph based on semantic similarity.
        
        Args:
            partial_text: Incomplete text to complete
            n_results: Number of completion suggestions to return
            context: Optional context filters (type, tags, etc.)
            
        Returns:
            List of completion suggestions with confidence scores
        """
        if not partial_text.strip():
            return []
        
        # Generate embedding for partial text
        partial_embedding = self.encoder.encode(partial_text).tolist()
        
        # Build query filters
        where_clause = {"word_count": {"$gte": len(partial_text.split())}}
        if context:
            if context.get("text_type"):
                where_clause["text_type"] = context["text_type"]
            if context.get("type"):
                where_clause["type"] = context["type"]
        
        try:
            # Find similar thought patterns
            results = self.collection.query(
                query_embeddings=[partial_embedding],
                n_results=min(n_results * 2, 50),  # Get more results to filter
                where=where_clause
            )
            
            # Extract completions from similar thoughts
            completions = []
            for i, doc in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i]
                distance = results['distances'][0][i]
                
                # Find where partial matches and extract remainder
                completion = self._extract_completion_from_match(partial_text, doc, metadata)
                if completion and completion.strip():
                    completions.append({
                        'text': completion,
                        'confidence': max(0, 1 - distance),  # Convert distance to confidence
                        'type': metadata.get('type', 'unknown'),
                        'source_text': doc,
                        'word_count': len(completion.split())
                    })
            
            # Sort by confidence and return top results
            completions.sort(key=lambda x: x['confidence'], reverse=True)
            return completions[:n_results]
            
        except Exception as e:
            logging.error(f"Error predicting completion: {e}")
            return []
    
    def _split_thoughts(self, text: str) -> List[str]:
        """Split text into semantic units (thoughts/sentences)."""
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Group sentences into thoughts
        thoughts = []
        current_thought = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            current_thought.append(sentence)
            
            # Check if this completes a thought
            if self._is_complete_thought(current_thought):
                thoughts.append(' '.join(current_thought))
                current_thought = []
        
        # Add remaining sentences as a thought
        if current_thought:
            thoughts.append(' '.join(current_thought))
        
        return [t for t in thoughts if t.strip()]
    
    def _is_complete_thought(self, sentences: List[str]) -> bool:
        """Determine if sentences form a complete thought."""
        if not sentences:
            return False
        
        text = ' '.join(sentences)
        word_count = len(text.split())
        
        # Heuristics for complete thoughts:
        # 1. Ends with conclusive punctuation
        # 2. Has reasonable length (5-100 words)
        # 3. Multiple sentences suggest paragraph break
        
        ends_conclusively = text.rstrip().endswith(('.', '!', '?', ':'))
        reasonable_length = 5 <= word_count <= 100
        multiple_sentences = len(sentences) > 1
        
        return ends_conclusively and reasonable_length and multiple_sentences
    
    def _classify_thought(self, thought: str) -> str:
        """Classify the type of thought/sentence."""
        thought_lower = thought.lower()
        
        # Simple classification based on patterns
        if thought.endswith('?'):
            return 'question'
        elif thought.endswith('!'):
            return 'exclamation'
        elif any(word in thought_lower for word in ['thanks', 'thank you', 'appreciate']):
            return 'gratitude'
        elif any(word in thought_lower for word in ['sorry', 'apologize', 'excuse']):
            return 'apology'
        elif thought_lower.startswith(('i think', 'i believe', 'in my opinion')):
            return 'opinion'
        elif any(word in thought_lower for word in ['will', 'would', 'could', 'should', 'plan to']):
            return 'intention'
        else:
            return 'statement'
    
    def _extract_completion(self, thought: str) -> str:
        """Extract the completion part of a thought for learning."""
        # For now, return the last part of the thought
        words = thought.split()
        if len(words) > 5:
            return ' '.join(words[-3:])  # Last 3 words
        return thought
    
    def _extract_completion_from_match(self, partial: str, full_text: str, metadata: Dict) -> str:
        """Extract completion from a matching document."""
        partial_lower = partial.lower().strip()
        full_lower = full_text.lower().strip()
        
        # Find where partial text appears in full text
        idx = full_lower.find(partial_lower)
        if idx != -1:
            # Extract the remainder after the partial match
            start_pos = idx + len(partial_lower)
            completion = full_text[start_pos:].strip()
            
            # Clean up completion
            if completion.startswith(','):
                completion = completion[1:].strip()
            
            return completion
        
        # If no direct match, try to find semantic completion
        # This is a simplified approach - could be enhanced
        words = full_text.split()
        if len(words) > len(partial.split()):
            # Return the latter part as completion
            partial_words = len(partial.split())
            return ' '.join(words[partial_words:])
        
        return ""
    
    def _should_store(self, text: str) -> bool:
        """Check if similar thought already exists to avoid duplicates."""
        if len(text.split()) < 3:
            return False
        
        try:
            # Check for very similar existing thoughts
            embedding = self.encoder.encode(text).tolist()
            similar = self.collection.query(
                query_embeddings=[embedding],
                n_results=1
            )
            
            if similar['distances'] and similar['distances'][0] and similar['distances'][0][0] < 0.1:
                # Very similar thought exists, don't store duplicate
                return False
            
            return True
        except Exception:
            # If query fails, err on the side of storing
            return True
    
    def cleanup_old_patterns(self, days: int = 90) -> int:
        """Remove patterns not accessed recently."""
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_iso = cutoff.isoformat()
        
        try:
            # Get old entries
            old_results = self.collection.get(
                where={"timestamp": {"$lt": cutoff_iso}}
            )
            
            if old_results['ids']:
                self.collection.delete(ids=old_results['ids'])
                return len(old_results['ids'])
            
            return 0
        except Exception as e:
            logging.error(f"Error cleaning up old patterns: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored semantic patterns."""
        try:
            count = self.collection.count()
            
            # Get type distribution
            all_data = self.collection.get()
            type_counts = {}
            word_counts = []
            
            for metadata in all_data.get('metadatas', []):
                thought_type = metadata.get('type', 'unknown')
                type_counts[thought_type] = type_counts.get(thought_type, 0) + 1
                word_counts.append(metadata.get('word_count', 0))
            
            avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
            
            return {
                'total_thoughts': count,
                'type_distribution': type_counts,
                'average_words_per_thought': round(avg_words, 1),
                'storage_path': self.db_path
            }
        except Exception as e:
            logging.error(f"Error getting stats: {e}")
            return {'error': str(e)}
