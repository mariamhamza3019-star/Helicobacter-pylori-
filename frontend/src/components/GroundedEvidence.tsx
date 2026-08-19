import React from 'react';
import { Quote, FileText } from 'lucide-react';
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
  if (!citations || citations.length === 0) {
    return null;
  }

  const groups = groupByExcerpt(citations);

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          <Quote size={17} style={{ color: '#2F6690' }} />
          Evidence &amp; Citations
        </h3>
        <span style={{ fontSize: '0.78rem', color: '#6B7280', fontWeight: 600 }}>
          {groups.length} Verbatim Passage{groups.length === 1 ? '' : 's'} · {citations.length} Grounded Reference{citations.length === 1 ? '' : 's'}
        </span>
      </div>
      <div className="card-body">
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
    </div>
  );
};
