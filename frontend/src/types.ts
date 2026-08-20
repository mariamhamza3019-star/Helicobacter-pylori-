export interface Citation {
  document: string;
  document_id: string;
  section: string;
  subsection?: string;
  chunk_id: string;
  page?: number;
  excerpt: string;
}

export interface RerankedDocument {
  rank: number;
  chunk_id: string;
  document: string;
  document_id: string;
  section: string;
  subsection?: string;
  page?: number;
  excerpt: string;
  text: string;
  score?: number;
  relevance?: number;
  raw_score?: number;
  bm25_score?: number;
  semantic_score?: number;
  rrf_score?: number;
  content_type?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface QueryResponse {
  recommendation: string;
  evidence: string[];
  citations: Citation[];
  reranked_documents: RerankedDocument[];
  confidence: 'high' | 'low';
  answer_status: 'answered' | 'insufficient_context' | 'greeting' | 'casual';
  refusal_reason?: string | null;
  latency_ms: number;
  pipeline_used: string;
  _meta?: {
    llm_called?: boolean;
    citation_warnings?: string[];
    top_score?: number;
    tone_warnings?: string[];
    reasoning_effort?: string;
    model?: string;
    skipped_retrieval?: boolean;
    intent?: string;
    fallback?: string;
    generation_failed?: boolean;
    both_llms_failed?: boolean;
  };
}

export interface GoldQuestion {
  id: string;
  q: string;
  expect_sections: string[];
}

export interface HealthStatus {
  status: string;
  index_loaded: boolean;
  num_chunks: number;
  has_api_key: boolean;
  pipeline: string;
}
