import React, { useState } from 'react';
import {
  Stethoscope,
  ShieldCheck,
  AlertTriangle,
  Clock,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Info,
  Sparkles,
  Layers
} from 'lucide-react';
import { QueryResponse } from '../types';
import { FormattedContent } from './FormattedContent';
import { PipelineDetails } from './PipelineDetails';

interface AssistantMessageProps {
  response: QueryResponse;
  timestamp: string;
  onSelectSuggestion?: (questionText: string) => void;
}

// Generate contextual follow-up questions based on the answer content
function getContextualSuggestions(queryOrAnswer: string): string[] {
  const text = queryOrAnswer.toLowerCase();
  if (text.includes('first-line') || text.includes('treatment-naive') || text.includes('bismuth') || text.includes('bqt')) {
    return [
      'What is the recommended duration for first-line BQT?',
      'Alternative first-line regimens if bismuth is unavailable?',
      'How to manage patients with penicillin allergy?'
    ];
  }
  if (text.includes('salvage') || text.includes('failure') || text.includes('persistent') || text.includes('treatment-experienced')) {
    return [
      'When is rifabutin triple therapy recommended?',
      'Role of high-dose PPI or PCAB dual therapy in salvage?',
      'When should levofloxacin triple therapy be used?'
    ];
  }
  if (text.includes('allergy') || text.includes('penicillin')) {
    return [
      'Is bismuth quadruple therapy safe with penicillin allergy?',
      'Can levofloxacin triple with metronidazole be used for allergy?',
      'When to confirm eradication after allergy regimen?'
    ];
  }
  if (text.includes('eradication') || text.includes('confirm') || text.includes('cure') || text.includes('test')) {
    return [
      'How long should PPI be stopped before urea breath test?',
      'Can serology be used for test-of-cure?',
      'What if persistent infection is confirmed after salvage?'
    ];
  }
  return [
    'What is the first-line treatment recommendation?',
    'Recommended salvage therapy after failure?',
    'When to confirm eradication?'
  ];
}

export const AssistantMessage: React.FC<AssistantMessageProps> = ({
  response,
  timestamp,
  onSelectSuggestion
}) => {
  const [showDetails, setShowDetails] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);

  const isAnswered = response.answer_status === 'answered';
  const isHighConfidence = response.confidence === 'high';
  const suggestions = getContextualSuggestions(response.recommendation);

  const handleCopy = () => {
    const textToCopy = `RECOMMENDATION:\n${response.recommendation}\n\nCITATIONS:\n${response.citations.map((c) => `${c.document} · ${c.section}${c.page ? ` · Page ${c.page}` : ''} · ${c.chunk_id}`).join('\n')}`;
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="assistant-message-wrapper">
      <div className="assistant-avatar">
        <Stethoscope size={18} />
      </div>

      <div className="assistant-card">
        {/* Card Header with Status & Timing */}
        <div className="assistant-card-header">
          <div className="status-and-title">
            <span className={`status-pill ${isAnswered ? 'status-answered' : 'status-insufficient'}`}>
              {isAnswered ? (
                <>
                  <ShieldCheck size={14} />
                  <span>{isHighConfidence ? 'Guideline Grounded' : 'Grounded Response'}</span>
                </>
              ) : (
                <>
                  <AlertTriangle size={14} />
                  <span>Insufficient Evidence · Refused</span>
                </>
              )}
            </span>
          </div>

          <div className="assistant-card-meta">
            <span className="timestamp-badge">
              <Clock size={12} /> {timestamp} · {response.latency_ms} ms
            </span>
            <button
              className="icon-action-btn"
              title="Copy recommendation"
              onClick={handleCopy}
            >
              {copied ? <Check size={14} color="#2E7D32" /> : <Copy size={14} />}
            </button>
          </div>
        </div>

        {/* 1. ANSWER-FIRST: SYNTHESIZED CLINICAL RECOMMENDATION */}
        <div className="answer-primary-block">
          <div className="answer-prose-text">
            <FormattedContent content={response.recommendation} />
          </div>

          {response.refusal_reason && (
            <div className="refusal-notice">
              <Info size={15} className="refusal-icon" />
              <span>{response.refusal_reason}</span>
            </div>
          )}
        </div>

        {/* 2. COLLAPSIBLE PIPELINE DETAILS TOGGLE */}
        <div className="pipeline-toggle-wrapper">
          <button
            className={`btn-pipeline-toggle ${showDetails ? 'is-expanded' : ''}`}
            onClick={() => setShowDetails(!showDetails)}
          >
            <Layers size={14} />
            <span>{showDetails ? 'Hide pipeline details' : 'View pipeline details (Retrieval · Reranking · Grounding · Citations)'}</span>
            {showDetails ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </button>
        </div>

        {/* 3. EXPANDED 4-STAGE PIPELINE DETAILS */}
        {showDetails && <PipelineDetails response={response} />}

        {/* 4. SUGGESTIONS ROW: PLACED BELOW DETAILS, NOT COMPETING WITH MAIN ANSWER */}
        {onSelectSuggestion && suggestions.length > 0 && (
          <div className="answer-followup-suggestions">
            <span className="followup-label">
              <Sparkles size={12} /> Follow-up Questions:
            </span>
            <div className="followup-chips-list">
              {suggestions.map((sug, idx) => (
                <button
                  key={idx}
                  className="followup-chip-btn"
                  onClick={() => onSelectSuggestion(sug)}
                >
                  {sug}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
