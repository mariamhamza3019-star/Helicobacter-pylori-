import React from 'react';
import { ShieldCheck, AlertTriangle, Clock, Layers, MessageCirclePlus } from 'lucide-react';
import { QueryResponse } from '../types';

interface RecommendationBoxProps {
  response: QueryResponse;
  onSelectFollowup?: (question: string) => void;
}

export const RecommendationBox: React.FC<RecommendationBoxProps> = ({ response, onSelectFollowup }) => {
  const isChitchat = response.answer_status === 'greeting' || response.answer_status === 'casual';
  const isAnswered = response.answer_status === 'answered';
  const isHighConfidence = response.confidence === 'high';

  return (
    <div className={`card recommendation-card ${isAnswered || isChitchat ? 'answered' : 'insufficient'}`}>
      <div className="card-body">
        <div className="rec-header">
          <div className={`rec-status-pill ${isAnswered || isChitchat ? 'answered' : 'insufficient'}`}>
            {isChitchat ? (
              <>
                <ShieldCheck size={16} />
                <span>H. pylori Assistant</span>
              </>
            ) : isAnswered ? (
              <>
                <ShieldCheck size={16} />
                <span>{isHighConfidence ? 'Guideline Answered · High Confidence' : 'Guideline Answered'}</span>
              </>
            ) : (
              <>
                <AlertTriangle size={16} />
                <span>Insufficient Guideline Evidence · Safely Refused</span>
              </>
            )}
          </div>

          <div style={{ display: 'flex', gap: '0.6rem', alignItems: 'center' }}>
            <span style={{ fontSize: '0.75rem', color: '#6B7280', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
              <Clock size={14} />
              {response.latency_ms} ms
            </span>
          </div>
        </div>

        <div className="rec-text">
          {response.recommendation}
        </div>

        {response.refusal_reason && (
          <div className="rec-refusal-box">
            <strong>Gating Notice:</strong> {response.refusal_reason}
          </div>
        )}

        {isAnswered && (
          <div className="meta-stats-row">
            <span><Layers size={13} style={{ display: 'inline', marginRight: 4 }} /> Pipeline: {response.pipeline_used}</span>
            <span>Citations: {response.citations.length}</span>
            <span>Reranked Candidates: {response.reranked_documents.length}</span>
            {response._meta?.top_score !== undefined && response._meta?.top_score !== null && (
              <span>Top Rerank Score: {Number(response._meta.top_score).toFixed(3)}</span>
            )}
          </div>
        )}
        {isAnswered && response.suggested_followups && response.suggested_followups.length > 0 && (
          <div className="followup-suggestions">
            <span className="followup-label">
              <MessageCirclePlus size={13} /> Related follow-ups
            </span>
            <div className="followup-chip-row">
              {response.suggested_followups.map((q, i) => (
                <button
                  key={i}
                  className="followup-chip"
                  onClick={() => onSelectFollowup && onSelectFollowup(q)}
                  type="button"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
