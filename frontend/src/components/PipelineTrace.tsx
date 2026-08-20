import React, { useState } from 'react';
import { Cpu, ChevronDown, ChevronUp, ShieldCheck, ShieldAlert, CheckCircle2, XCircle } from 'lucide-react';
import { QueryResponse } from '../types';

interface PipelineTraceProps {
  response: QueryResponse;
}

const PIPELINE_LABELS: Record<string, string> = {
  rrf_rerank: 'Hybrid RRF + MedCPT Cross-Encoder Rerank',
  dense_rerank: 'Dense FAISS + MedCPT Rerank',
  rrf: 'RRF Hybrid (No Rerank)',
  minmax: 'MinMax 70/30 Hybrid',
  bm25: 'BM25 Lexical Only',
};

export const PipelineTrace: React.FC<PipelineTraceProps> = ({ response }) => {
  const [open, setOpen] = useState(false);

  const meta = response._meta || {};
  const citationWarnings = meta.citation_warnings || [];
  const toneWarnings = meta.tone_warnings || [];
  const gatePassed = response.answer_status === 'answered';
  const citationsKept = response.citations.length;
  const citationsStripped = citationWarnings.length;

  return (
    <div className="card" style={{ background: '#fafbfc' }}>
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
          <Cpu size={16} />
          Pipeline &amp; Safety Trace
        </span>
        {open ? <ChevronUp size={16} color="#6B7280" /> : <ChevronDown size={16} color="#6B7280" />}
      </button>

      {open && (
        <div className="card-body" style={{ paddingTop: 0, display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <div className="trace-row">
            <span className="trace-label">Retrieval pipeline</span>
            <span className="trace-value">{PIPELINE_LABELS[response.pipeline_used] || response.pipeline_used}</span>
          </div>

          <div className="trace-row">
            <span className="trace-label">Relevance gate</span>
            <span className="trace-value" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: gatePassed ? '#2E7D32' : '#D97706' }}>
              {gatePassed ? <CheckCircle2 size={14} /> : <ShieldAlert size={14} />}
              {gatePassed ? 'Passed' : 'Refused — below threshold'}
              {meta.top_score !== undefined && meta.top_score !== null && (
                <span style={{ color: '#6B7280', fontWeight: 500 }}>(top score {Number(meta.top_score).toFixed(3)})</span>
              )}
            </span>
          </div>

          <div className="trace-row">
            <span className="trace-label">Citation verification</span>
            <span className="trace-value" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={14} color="#2E7D32" />
              {citationsKept} kept
              {citationsStripped > 0 && (
                <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', color: '#D97706' }}>
                  <XCircle size={14} /> {citationsStripped} stripped (failed excerpt-grounding match)
                </span>
              )}
            </span>
          </div>

          <div className="trace-row">
            <span className="trace-label">Tone guardrail</span>
            <span
              className="trace-value"
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.35rem',
                color: !gatePassed ? '#6B7280' : toneWarnings.length ? '#D97706' : '#2E7D32',
              }}
            >
              {!gatePassed ? (
                <ShieldCheck size={14} color="#6B7280" />
              ) : toneWarnings.length ? (
                <ShieldAlert size={14} />
              ) : (
                <ShieldCheck size={14} />
              )}
              {!gatePassed
                ? 'N/A — Query refused'
                : toneWarnings.length
                ? `${toneWarnings.length} flagged phrase(s)`
                : 'Passed — no directive language detected'}
            </span>
          </div>

          {gatePassed && toneWarnings.length > 0 && (
            <div className="trace-warning-list">
              {toneWarnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          {citationsStripped > 0 && (
            <div className="trace-warning-list">
              {citationWarnings.map((w, i) => (
                <div key={i}>{w}</div>
              ))}
            </div>
          )}

          <div className="trace-row">
            <span className="trace-label">Model</span>
            <span className="trace-value">{meta.model || 'openai/gpt-oss-120b'} · reasoning: {meta.reasoning_effort || 'low'}</span>
          </div>

          <div className="trace-row">
            <span className="trace-label">Latency</span>
            <span className="trace-value">{response.latency_ms} ms</span>
          </div>
        </div>
      )}
    </div>
  );
};