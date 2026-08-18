import React from 'react';
import { BookmarkCheck } from 'lucide-react';
import { Citation } from '../types';

interface CitationsListProps {
  citations: Citation[];
}

export const CitationsList: React.FC<CitationsListProps> = ({ citations }) => {
  if (!citations || citations.length === 0) {
    return null;
  }

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          <BookmarkCheck size={17} style={{ color: '#2F6690' }} />
          Verified Citations
        </h3>
        <span style={{ fontSize: '0.78rem', color: '#6B7280', fontWeight: 600 }}>
          {citations.length} Grounded References
        </span>
      </div>
      <div className="card-body">
        <div className="citations-grid">
          {citations.map((cite, idx) => (
            <div key={idx} className="citation-box">
              <div>
                <div className="citation-header">
                  <span>#{idx + 1} {cite.section}</span>
                </div>
                {cite.subsection && (
                  <div style={{ fontSize: '0.76rem', color: '#2F6690', fontWeight: 600, marginBottom: '0.35rem' }}>
                    / {cite.subsection}
                  </div>
                )}
                {cite.excerpt && (
                  <div className="citation-excerpt">
                    "{cite.excerpt}"
                  </div>
                )}
              </div>
              <div className="citation-meta">
                <span>{cite.document}</span>
                <span style={{ display: 'block', marginTop: '2px', color: '#1E3A5F', fontWeight: 600 }}>
                  {cite.chunk_id} {cite.page ? `· Page ${cite.page}` : ''}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
