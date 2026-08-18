import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { QuerySection } from './components/QuerySection';
import { RecommendationBox } from './components/RecommendationBox';
import { EvidenceSection } from './components/EvidenceSection';
import { CitationsList } from './components/CitationsList';
import { RerankedDocs } from './components/RerankedDocs';
import { GoldWorkbench } from './components/GoldWorkbench';
import { HealthStatus, QueryResponse } from './types';
import { AlertCircle } from 'lucide-react';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [activeTab, setActiveTab] = useState<'search' | 'gold'>('search');
  const [query, setQuery] = useState<string>(
    'What is the preferred first-line treatment for treatment-naive H. pylori when susceptibility is unknown?'
  );
  const [topK, setTopK] = useState<number>(5);
  const [pipeline, setPipeline] = useState<string>('rrf_rerank');
  const [relevanceThreshold, setRelevanceThreshold] = useState<number>(0.35);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<QueryResponse | null>(null);

  useEffect(() => {
    // Initial health check
    fetch('/health')
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch((err) => {
        console.warn('Backend not ready yet:', err);
      });
  }, []);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          top_k: topK,
          pipeline,
          relevance_threshold: relevanceThreshold,
          use_llm: true,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Unknown error occurred' }));
        throw new Error(errorData.detail || `HTTP Error ${res.status}`);
      }

      const data: QueryResponse = await res.json();
      setResponse(data);
    } catch (err: any) {
      console.error('Query error:', err);
      setError(err.message || 'Failed to retrieve guidelines. Ensure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectGoldQuestion = (questionText: string) => {
    setQuery(questionText);
    setActiveTab('search');
  };

  return (
    <div className="app-container">
      <Header
        health={health}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />

      <main className="main-content">
        {activeTab === 'search' && (
          <>
            <QuerySection
              query={query}
              setQuery={setQuery}
              onSearch={handleSearch}
              loading={loading}
              topK={topK}
              setTopK={setTopK}
              pipeline={pipeline}
              setPipeline={setPipeline}
              relevanceThreshold={relevanceThreshold}
              setRelevanceThreshold={setRelevanceThreshold}
            />

            {error && (
              <div className="card" style={{ borderColor: '#C62828', background: '#ffebee' }}>
                <div className="card-body" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#C62828' }}>
                  <AlertCircle size={20} />
                  <div>
                    <strong>Query Error:</strong> {error}
                  </div>
                </div>
              </div>
            )}

            {response && (
              <>
                <RecommendationBox response={response} />
                <EvidenceSection evidence={response.evidence} citations={response.citations} />
                <CitationsList citations={response.citations} />
                <RerankedDocs documents={response.reranked_documents} />
              </>
            )}
          </>
        )}

        {activeTab === 'gold' && (
          <GoldWorkbench onSelectQuestion={handleSelectGoldQuestion} />
        )}
      </main>
    </div>
  );
};

export default App;
