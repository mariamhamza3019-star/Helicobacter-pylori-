import React from 'react';
import { Search, SlidersHorizontal } from 'lucide-react';

interface QuerySectionProps {
  query: string;
  setQuery: (q: string) => void;
  onSearch: () => void;
  loading: boolean;
  topK: number;
  setTopK: (k: number) => void;
  pipeline: string;
  setPipeline: (p: string) => void;
  relevanceThreshold: number;
  setRelevanceThreshold: (t: number) => void;
}

const SAMPLE_QUERIES = [
  "What is the preferred first-line therapy for treatment-naive patients when antibiotic susceptibility is unknown?",
  "How should H. pylori be managed in patients with a penicillin allergy?",
  "How many days should bismuth quadruple therapy (BQT) be given for?",
  "What is the recommended H. pylori regimen in pregnancy?",
  "How is H. pylori classified regarding gastric cancer risk?",
  "Should all patients undergo post-treatment testing to confirm eradication?"
];

export const QuerySection: React.FC<QuerySectionProps> = ({
  query,
  setQuery,
  onSearch,
  loading,
  topK,
  setTopK,
  pipeline,
  setPipeline,
  relevanceThreshold,
  setRelevanceThreshold,
}) => {
  const [showSettings, setShowSettings] = React.useState(false);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      onSearch();
    }
  };

  return (
    <div className="card query-box-card">
      <div className="card-body">
        <div className="query-input-wrapper">
          <input
            type="text"
            className="query-input"
            placeholder="Ask a clinical question regarding H. pylori diagnosis, therapy, or management..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            className="btn-primary"
            onClick={onSearch}
            disabled={loading || !query.trim()}
          >
            {loading ? (
              <>
                <span className="spinner"></span>
                <span>Retrieving & Reranking...</span>
              </>
            ) : (
              <>
                <Search size={18} />
                <span>Ask Guideline</span>
              </>
            )}
          </button>
        </div>

        <div className="query-chips-bar">
          <span className="chips-label">Clinical Examples:</span>
          {SAMPLE_QUERIES.map((sample, idx) => (
            <button
              key={idx}
              className="query-chip"
              onClick={() => {
                setQuery(sample);
              }}
              disabled={loading}
            >
              {sample.length > 48 ? sample.substring(0, 48) + "..." : sample}
            </button>
          ))}
          <button
            className="query-chip"
            style={{ marginLeft: 'auto', background: showSettings ? '#e2e8f0' : '#f8fafc' }}
            onClick={() => setShowSettings(!showSettings)}
          >
            <SlidersHorizontal size={13} style={{ display: 'inline', marginRight: 4 }} />
            Pipeline Settings
          </button>
        </div>

        {showSettings && (
          <div className="settings-bar">
            <div className="setting-item">
              <label>Pipeline:</label>
              <select
                className="setting-select"
                value={pipeline}
                onChange={(e) => setPipeline(e.target.value)}
                disabled={loading}
              >
                <option value="rrf_rerank">RRF + MedCPT Cross-Encoder (Shipping Stack)</option>
                <option value="dense_rerank">Dense FAISS + MedCPT Rerank</option>
                <option value="rrf">RRF Hybrid (No Rerank)</option>
                <option value="minmax">MinMax 70/30 Hybrid</option>
                <option value="bm25">BM25 Lexical Only</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Top K Chunks:</label>
              <select
                className="setting-select"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={loading}
              >
                <option value={3}>3 Chunks</option>
                <option value={5}>5 Chunks (Default)</option>
                <option value={8}>8 Chunks</option>
                <option value={10}>10 Chunks</option>
              </select>
            </div>

            <div className="setting-item">
              <label>Min Score Gate:</label>
              <input
                type="number"
                className="setting-input"
                style={{ width: 70 }}
                step="0.05"
                min="0.0"
                max="1.0"
                value={relevanceThreshold}
                onChange={(e) => setRelevanceThreshold(Number(e.target.value))}
                disabled={loading}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
