# embedding_generator.py
# Task 4: Generating Embeddings for Chunks

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Allow imports when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import CHUNKS_JSON, EMBEDDINGS_JSON

class EmbeddingGenerator:
    def __init__(self, model_name='pritamdeka/S-BioBert-snli-multinli-stsb'):
        print(f"🔄 تحميل نموذج {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print("✅ تم تحميل النموذج بنجاح!")
    
    def load_chunks(self, input_file):
        """تحميل الـ chunks من ملف JSON"""
        print(f"📂 تحميل الملف: {input_file}")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # محاولة قراءة كـ JSON عادي
            try:
                data = json.loads(content)
                if isinstance(data, dict) and 'chunks' in data:
                    chunks = data['chunks']
                else:
                    chunks = data
            except json.JSONDecodeError:
                # لو مش JSON عادي، جربه كـ JSON Lines
                lines = content.strip().split('\n')
                chunks = [json.loads(line) for line in lines if line.strip()]
        
        print(f"✅ تم تحميل {len(chunks)} chunk")
        return chunks
    
    def generate_embeddings(self, chunks):
        """توليد الـ embeddings لكل chunk"""
        print("🔄 جاري توليد الـ embeddings...")
        
        texts = [chunk['text'] for chunk in chunks]
        
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        print(f"✅ تم توليد {len(embeddings)} embedding")
        print(f"📐 بُعد كل vector: {embeddings.shape[1]}")
        
        # إضافة embeddings للـ chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
        
        return chunks, embeddings
    
    def save_output(self, chunks, embeddings, output_file):
        """حفظ الـ chunks مع الـ embeddings"""
        output_data = {
            "metadata": {
                "embedding_model": self.model_name,
                "embedding_dimension": embeddings.shape[1],
                "total_chunks": len(chunks),
                "created_at": datetime.now().isoformat()
            },
            "chunks": chunks
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ تم حفظ الملف: {output_file}")
        return output_data
    
    def run_pipeline(self, input_file, output_file):
        """تشغيل الـ pipeline بالكامل"""
        print("="*60)
        print("🚀 TASK 4: Embedding Generator")
        print("="*60)
        
        chunks = self.load_chunks(input_file)
        chunks, embeddings = self.generate_embeddings(chunks)
        
        print(f"\n📊 إحصائيات:")
        print(f"   - عدد الـ chunks: {len(chunks)}")
        print(f"   - بُعد الـ embeddings: {embeddings.shape[1]}")
        
        output = self.save_output(chunks, embeddings, output_file)
        
        print("\n" + "="*60)
        print("✅ Task 4 completed successfully!")
        print("="*60)
        
        return output

def main():
    INPUT_FILE = CHUNKS_JSON
    OUTPUT_FILE = EMBEDDINGS_JSON

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found — run ingestion first:")
        print("  python run_pipeline.py")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    generator = EmbeddingGenerator()
    generator.run_pipeline(str(INPUT_FILE), str(OUTPUT_FILE))

if __name__ == "__main__":
    main()