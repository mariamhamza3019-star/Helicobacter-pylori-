import React from 'react';
import { Quote, FileText } from 'lucide-react';
import { Citation } from '../types';

interface EvidenceSectionProps {
  evidence: string[];
  citations: Citation[];
}

export const EvidenceSection: React.FC<EvidenceSectionProps> = ({ evidence, citations }) => {
  if (!evidence || evidence.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          <Quote size={17} style={{ color: '#2F6690' }} />
          Evidence / Excerpts
        </h3>
        <span style={{ fontSize: '0.78rem', color: '#6B7280', fontWeight: 600 }}>
          {evidence.length} Verbatim Guideline Passages
        </span>
      </div>
      <div className="card-body">
        {evidence.map((excerpt, idx) => {
          const matchingCite = citations[idx] || citations[0];
          return (
            <div key={idx} className="evidence-card">
              <div className="evidence-text">
                "{excerpt}"
              </div>
              {matchingCite && (
                <div className="evidence-source-tag">
                  <FileText size={13} />
                  <span>
                    {matchingCite.document} · <strong>{matchingCite.section}</strong>
                    {matchingCite.page ? ` · Page ${matchingCite.page}` : ''}
                    {matchingCite.chunk_id ? ` · (${matchingCite.chunk_id})` : ''}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
