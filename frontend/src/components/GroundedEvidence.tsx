import React, { useState } from 'react';
import { Quote, FileText, ChevronDown, ChevronUp } from 'lucide-react';
import { Citation } from '../types';

interface GroundedEvidenceProps {
  citations: Citation[];
}

interface EvidenceGroup {
  excerpt: string;
  sources: Citation[];
}

/**
 * Groups citations that share the same (or near-identical) excerpt text.
 * The ACG guideline restates some facts verbatim across multiple pages
 * (e.g. prevalence numbers appear on both p.3 and p.5), so the same quote
 * legitimately gets cited from more than one chunk. Rather than showing
 * that quote twice, we show it once with every source it came from.
 */
function groupByExcerpt(citations: Citation[]): EvidenceGroup[] {
  const groups: EvidenceGroup[] = [];
  const indexByKey = new Map<string, number>();

  for (const cite of citations) {
    const key = (cite.excerpt || '').trim().toLowerCase();
    if (!key) continue;

    const existingIdx = indexByKey.get(key);
    if (existingIdx !== undefined) {
      groups[existingIdx].sources.push(cite);
    } else {
      indexByKey.set(key, groups.length);
      groups.push({ excerpt: cite.excerpt.trim(), sources: [cite] });
    }
  }

  return groups;
}

export const GroundedEvidence: React.FC<GroundedEvidenceProps> = ({ citations }) => {
  const [open, setOpen] = useState(false);

  if (!citations || citations.length === 0) {
    return null;
  }

  const groups = groupByExcerpt(citations);

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
          <Quote size={17} />
          Evidence &amp; Citations
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <span style={{ fontSize: '0.78rem', color: '#6B7280', fontWeight: 600 }}>
            {groups.length} Verbatim Passage{groups.length === 1 ? '' : 's'} · {citations.length} Grounded Reference{citations.length === 1 ? '' : 's'}
          </span>
          {open ? <ChevronUp size={16} color="#6B7280" /> : <ChevronDown size={16} color="#6B7280" />}
        </span>
      </button>

      {open && (
        <div className="card-body" style={{ paddingTop: 0 }}>
          {groups.map((group, idx) => (
            <div key={idx} className="evidence-card">
              <div className="evidence-text">"{group.excerpt}"</div>
              <div className="evidence-source-list">
                {group.sources.map((src, sIdx) => (
                  <div key={sIdx} className="evidence-source-badge">
                    <FileText size={12} />
                    <span>
                      {src.section}
                      {src.subsection ? ` / ${src.subsection}` : ''}
                      {src.page ? ` · Page ${src.page}` : ''}
                      {src.chunk_id ? ` · (${src.chunk_id})` : ''}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};