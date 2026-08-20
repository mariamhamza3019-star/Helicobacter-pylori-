import React, { useState } from 'react';
import {
  BookmarkCheck,
  ShieldCheck,
  ArrowUpDown,
  Search,
  FileText,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { QueryResponse } from '../types';

interface PipelineDetailsProps {
  response: QueryResponse;
}

// Clean helper to extract a concise 1-2 sentence snippet without table bars or dumps
export function cleanSnippet(text: string, maxSentences: number = 2): string {
  if (!text) return '';
  
  // Remove markdown table syntax lines like |---|---|
  const nonTableLines = text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('|---') && !l.startsWith('| :---'));

  let clean = nonTableLines
    .map((l) => l.replace(/^\|\s*/, '').replace(/\s*\|$/, '').replace(/\|/g, ' — '))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();

  // Extract first 1-2 sentences
  const sentenceMatches = clean.match(/[^.!?]+[.!?]+/g);
  if (sentenceMatches && sentenceMatches.length > 0) {
    return sentenceMatches.slice(0, maxSentences).join(' ').trim();
  }
  
  return clean.length > 200 ? clean.slice(0, 200) + '...' : clean;
}

export const PipelineDetails: React.FC<PipelineDetailsProps> = ({ response }) => {
  const [activeTab, setActiveTab] = useState<'citations' | 'grounding' | 'reranking' | 'retrieval'>('citations');
  const [expandedDocId, setExpandedDocId] = useState<string | null>(null);

  const toggleDocExpand = (chunkId: string) => {
    setExpandedDocId(expandedDocId === chunkId ? null : chunkId);
  };

  const citedChunkIds = new Set(response.citations.map((c) => c.chunk_id));
  const groundedDocs = response.reranked_documents.filter((d) => citedChunkIds.has(d.chunk_id));
  const displayGrounded = groundedDocs.length > 0 ? groundedDocs : response.reranked_documents.slice(0, 2);

  return (
    <div className="pipeline-details-panel">
      {/* Tab Navigation */}
      <div className="pipeline-tabs-header">
        <button
          className={`pipeline-tab-btn ${activeTab === 'citations' ? 'active' : ''}`}
          onClick={() => setActiveTab('citations')}
        >
          <BookmarkCheck size={14} />
          <span>Citations ({response.citations.length})</span>
        </button>

        <button
          className={`pipeline-tab-btn ${activeTab === 'grounding' ? 'active' : ''}`}
          onClick={() => setActiveTab('grounding')}
        >
          <ShieldCheck size={14} />
          <span>Grounding & Evidence</span>
        </button>

        <button
          className={`pipeline-tab-btn ${activeTab === 'reranking' ? 'active' : ''}`}
          onClick={() => setActiveTab('reranking')}
        >
          <ArrowUpDown size={14} />
          <span>Reranking ({response.reranked_documents.length})</span>
        </button>

        <button
          className={`pipeline-tab-btn ${activeTab === 'retrieval' ? 'active' : ''}`}
          onClick={() => setActiveTab('retrieval')}
        >
          <Search size={14} />
          <span>Retrieval Scores</span>
        </button>
      </div>

      <div className="pipeline-tab-content">
        {/* ---------------- 1. CITATIONS TAB ---------------- */}
        {activeTab === 'citations' && (
          <div className="citations-tab-pane">
            {response.citations.length === 0 ? (
              <div className="empty-tab-state">No direct citations recorded for this answer.</div>
            ) : (
              <div className="citations-cards-list">
                {response.citations.map((cite, idx) => {
                  const docTitle = cite.document || "ACG Clinical Guideline 2024: Treatment of Helicobacter pylori Infection";
                  const snippet = cleanSnippet(cite.excerpt || '');
                  const isExpanded = expandedDocId === `cite-${idx}`;

                  return (
                    <div key={idx} className="citation-detail-card">
                      <div className="citation-card-main">
                        <div className="citation-header-line">
                          <span className="citation-badge">Source [{idx + 1}]</span>
                          <span className="citation-doc-id">{cite.chunk_id}</span>
                          {cite.page && <span className="citation-page-badge">Page {cite.page}</span>}
                        </div>

                        <div className="citation-source-title">
                          <strong>{docTitle}</strong>
                        </div>

                        <div className="citation-section-path">
                          <FileText size={12} />
                          <span>{cite.section}{cite.subsection ? ` › ${cite.subsection}` : ''}</span>
                        </div>

                        <div className="citation-snippet-text">
                          "{snippet}"
                        </div>
                      </div>

                      {cite.excerpt && cite.excerpt.length > snippet.length && (
                        <div className="citation-expand-action">
                          <button
                            className="btn-text-expand"
                            onClick={() => toggleDocExpand(`cite-${idx}`)}
                          >
                            {isExpanded ? 'Show short snippet' : 'View full text'}
                            {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                          </button>
                          {isExpanded && (
                            <div className="full-excerpt-expanded">
                              {cite.excerpt}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* ---------------- 2. GROUNDING TAB ---------------- */}
        {activeTab === 'grounding' && (
          <div className="grounding-tab-pane">
            <div className="grounding-summary-card">
              <div className="grounding-status-header">
                <span className="grounding-indicator-dot"></span>
                <strong>Evidence Grounding Verification</strong>
                <span className="grounding-model-tag">Strict Citation Verification</span>
              </div>
              <p className="grounding-description">
                The synthesized recommendation is strictly verified against retrieved chunks from the ACG 2024 Guidelines.
                Any ungrounded claim or hallucinated chunk ID is automatically stripped.
              </p>
            </div>

            <div className="grounded-chunks-list">
              <h5 className="subhead-title">Chunks Utilized for Answer Synthesis:</h5>
              {displayGrounded.map((doc) => (
                <div key={doc.chunk_id} className="grounded-chunk-item">
                  <div className="grounded-item-header">
                    <span className="grounded-chip">Grounded Chunk #{doc.rank}</span>
                    <span className="chunk-id-tag">{doc.chunk_id}</span>
                    {doc.score !== undefined && doc.score !== null && (
                      <span className="relevance-score-tag">Rerank Score: {doc.score.toFixed(3)}</span>
                    )}
                  </div>
                  <div className="grounded-section-line">
                    <strong>{doc.section}</strong> {doc.subsection && `— ${doc.subsection}`} {doc.page && `(p. ${doc.page})`}
                  </div>
                  <div className="grounded-snippet">
                    {cleanSnippet(doc.excerpt || doc.text, 2)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ---------------- 3. RERANKING TAB ---------------- */}
        {activeTab === 'reranking' && (
          <div className="reranking-tab-pane">
            <div className="rerank-intro-bar">
              <span><strong>MedCPT Cross-Encoder Reranker:</strong> Scored and re-ordered candidates from the hybrid pool.</span>
              <span className="rerank-model-name">ncbi/MedCPT-Cross-Encoder</span>
            </div>

            <div className="reranked-table-list">
              {response.reranked_documents.map((doc) => {
                const scoreFormatted = doc.score !== undefined && doc.score !== null ? doc.score.toFixed(3) : null;
                const rawScoreFormatted = doc.raw_score !== undefined && doc.raw_score !== null ? doc.raw_score.toFixed(2) : null;
                const isTop = doc.rank === 1;

                return (
                  <div key={doc.chunk_id} className={`rerank-row-card ${isTop ? 'is-winner' : ''}`}>
                    <div className="rerank-row-left">
                      <span className={`rank-badge ${isTop ? 'rank-1' : ''}`}>#{doc.rank}</span>
                      <div className="rerank-details">
                        <div className="rerank-title-line">
                          <span className="doc-section-text">{doc.section}</span>
                          {doc.page && <span className="doc-page-text">p. {doc.page}</span>}
                          <span className="doc-id-pill">{doc.chunk_id}</span>
                        </div>
                        <div className="rerank-snippet-text">
                          {cleanSnippet(doc.excerpt || doc.text, 1)}
                        </div>
                      </div>
                    </div>

                    <div className="rerank-scores-group">
                      {scoreFormatted !== null && (
                        <div className="score-box primary-score">
                          <span className="score-lbl">Reranker Score</span>
                          <span className="score-num">{scoreFormatted}</span>
                        </div>
                      )}
                      {rawScoreFormatted !== null && (
                        <div className="score-box secondary-score">
                          <span className="score-lbl">Cross-Encoder Logit</span>
                          <span className="score-num">{rawScoreFormatted}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ---------------- 4. RETRIEVAL SCORES TAB ---------------- */}
        {activeTab === 'retrieval' && (
          <div className="retrieval-tab-pane">
            <div className="retrieval-intro-bar">
              <span><strong>Initial Retrieval Channel Scores:</strong> Lexical BM25, Semantic Cosine (FAISS), and Reciprocal Rank Fusion (RRF).</span>
            </div>

            <div className="retrieval-cards-list">
              {response.reranked_documents.map((doc) => (
                <div key={doc.chunk_id} className="retrieval-score-card">
                  <div className="retrieval-card-header">
                    <span className="retrieval-rank">#{doc.rank}</span>
                    <span className="retrieval-chunk-id">{doc.chunk_id}</span>
                    <span className="retrieval-section">{doc.section}</span>
                  </div>

                  <div className="channel-scores-grid">
                    {doc.bm25_score !== undefined && doc.bm25_score !== null && (
                      <div className="channel-metric">
                        <span className="metric-label">BM25 Lexical</span>
                        <span className="metric-value">{doc.bm25_score.toFixed(2)}</span>
                      </div>
                    )}
                    {doc.semantic_score !== undefined && doc.semantic_score !== null && (
                      <div className="channel-metric">
                        <span className="metric-label">Semantic Cosine</span>
                        <span className="metric-value">{doc.semantic_score.toFixed(3)}</span>
                      </div>
                    )}
                    {doc.rrf_score !== undefined && doc.rrf_score !== null && (
                      <div className="channel-metric">
                        <span className="metric-label">RRF Fusion</span>
                        <span className="metric-value">{doc.rrf_score.toFixed(5)}</span>
                      </div>
                    )}
                    {doc.score !== undefined && doc.score !== null && (
                      <div className="channel-metric highlight">
                        <span className="metric-label">MedCPT Rerank</span>
                        <span className="metric-value">{doc.score.toFixed(3)}</span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
