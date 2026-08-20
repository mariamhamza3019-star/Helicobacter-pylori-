import React, { useState } from 'react';
import { Layers, ChevronDown, ChevronUp } from 'lucide-react';
import { RerankedDocument } from '../types';

interface RerankedDocsProps {
  documents: RerankedDocument[];
}

export const RerankedDocs: React.FC<RerankedDocsProps> = ({ documents }) => {
  const [open, setOpen] = useState(false);
  const [expandedChunkId, setExpandedChunkId] = useState<string | null>(null);

  if (!documents || documents.length === 0) {
    return null;
  }

  const toggleExpand = (chunkId: string) => {
    setExpandedChunkId(expandedChunkId === chunkId ? null : chunkId);
  };

  return (
    <div className="card">
      <button
        onClick={() => setOpen(!open)}
        style={{
          width: '100%',
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          padding: '0.85rem 1.1rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.85rem', color: '#1E3A5F' }}>
          <Layers size={17} />
          Reranked Documents
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '0.78rem', color: '#6B7280', fontWeight: 600 }}>
            Top {documents.length} Candidate Chunks
          </span>
          {open ? <ChevronUp size={16} color="#6B7280" /> : <ChevronDown size={16} color="#6B7280" />}
        </span>
      </button>

      {open && (
        <div className="card-body" style={{ paddingTop: 0 }}>
          <div className="rerank-list">
            {documents.map((doc) => {
              const isExpanded = expandedChunkId === doc.chunk_id;
              const scoreFormatted = doc.score !== undefined && doc.score !== null
                ? doc.score.toFixed(2)
                : (doc.relevance !== undefined && doc.relevance !== null ? doc.relevance.toFixed(2) : 'N/A');
              const isHighScore = (doc.score ?? 0) >= 0.70;

              return (
                <div
                  key={doc.chunk_id}
                  className={`rerank-card ${doc.rank === 1 ? 'top-rank' : ''}`}
                >
                  <div className="rerank-top-row">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                      <span className="rank-badge">#{doc.rank}</span>
                      <span className={`relevance-score-badge ${isHighScore ? 'high' : ''}`}>
                        Relevance: {scoreFormatted}
                      </span>
                    </div>

                    <button
                      onClick={() => toggleExpand(doc.chunk_id)}
                      style={{
                        background: 'none',
                        border: 'none',
                        fontSize: '0.76rem',
                        color: '#2F6690',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        fontWeight: 600
                      }}
                    >
                      {isExpanded ? (
                        <>Hide Details <ChevronUp size={14} /></>
                      ) : (
                        <>Channel Breakdown <ChevronDown size={14} /></>
                      )}
                    </button>
                  </div>

                  <div className="section-title">
                    Section: {doc.section}
                    {doc.subsection && (
                      <span style={{ color: '#2F6690', fontWeight: 600 }}> / {doc.subsection}</span>
                    )}
                  </div>

                  <div className="rerank-excerpt">
                    <strong>Excerpt:</strong> "{doc.excerpt || doc.text}"
                  </div>

                  <div className="rerank-source-info">
                    <span>
                      <strong>Source:</strong> {doc.document}
                      {doc.page ? ` · Page ${doc.page}` : ''}
                      {` · ${doc.chunk_id}`}
                    </span>
                    {doc.content_type && (
                      <span style={{ textTransform: 'uppercase', fontSize: '0.7rem' }}>
                        {doc.content_type}
                      </span>
                    )}
                  </div>

                  {isExpanded && (
                    <div style={{
                      marginTop: '0.75rem',
                      paddingTop: '0.75rem',
                      borderTop: '1px dashed #D9E1E8',
                      display: 'flex',
                      flexWrap: 'wrap',
                      gap: '1rem',
                      fontSize: '0.74rem',
                      fontFamily: 'var(--font-mono)',
                      color: '#4B5563',
                      background: '#F8FAFC',
                      padding: '0.5rem 0.75rem',
                      borderRadius: '4px'
                    }}>
                      <span>Actual Rerank Score (Norm): <strong>{doc.score !== undefined ? doc.score.toFixed(4) : 'N/A'}</strong></span>
                      {doc.raw_score !== undefined && doc.raw_score !== null && (
                        <span>Raw Cross-Encoder Logit: <strong>{doc.raw_score.toFixed(3)}</strong></span>
                      )}
                      {doc.bm25_score !== undefined && doc.bm25_score !== null && (
                        <span>BM25 Score: <strong>{doc.bm25_score.toFixed(2)}</strong></span>
                      )}
                      {doc.semantic_score !== undefined && doc.semantic_score !== null && (
                        <span>Semantic Sim: <strong>{doc.semantic_score.toFixed(3)}</strong></span>
                      )}
                      {doc.rrf_score !== undefined && doc.rrf_score !== null && (
                        <span>RRF Score: <strong>{doc.rrf_score.toFixed(5)}</strong></span>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};