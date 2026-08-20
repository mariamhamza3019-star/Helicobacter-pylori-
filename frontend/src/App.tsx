import React, { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { QuerySection } from './components/QuerySection';
import { RecommendationBox } from './components/RecommendationBox';
import { GroundedEvidence } from './components/GroundedEvidence';
import { PipelineTrace } from './components/PipelineTrace';
import { RerankedDocs } from './components/RerankedDocs';
import { GoldWorkbench } from './components/GoldWorkbench';
import { HealthStatus, QueryResponse, ChatMessage } from './types';
import { AlertCircle, User } from 'lucide-react';

interface ChatTurn {
  id: number;
  query: string;
  response: QueryResponse | null;
  error: string | null;
  loading: boolean;
}

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [activeTab, setActiveTab] = useState<'search' | 'gold'>('search');
  const [query, setQuery] = useState<string>('');
  const [topK, setTopK] = useState<number>(5);
  const [pipeline, setPipeline] = useState<string>('rrf_rerank');
  const [relevanceThreshold, setRelevanceThreshold] = useState<number>(0.35);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const nextId = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initial health check
    fetch('/health')
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch((err) => {
        console.warn('Backend not ready yet:', err);
      });
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const loading = turns.some((t) => t.loading);

  const handleSearch = async () => {
    const q = query.trim();
    if (!q || loading) return;

    const id = nextId.current++;
    // Simplified scope: only the immediately previous completed turn is
    // used as context for a follow-up, not the full growing conversation.
    const priorTurns = turns.filter((t) => t.response).slice(-1);

    setTurns((prev) => [...prev, { id, query: q, response: null, error: null, loading: true }]);
    setQuery('');

    const history: ChatMessage[] = priorTurns.flatMap((t) => [
      { role: 'user', content: t.query },
      { role: 'assistant', content: t.response!.recommendation },
    ]);

    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: q,
          top_k: topK,
          pipeline,
          relevance_threshold: relevanceThreshold,
          use_llm: true,
          history,
        }),
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: 'Unknown error occurred' }));
        throw new Error(errorData.detail || `HTTP Error ${res.status}`);
      }

      const data: QueryResponse = await res.json();
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, response: data, loading: false } : t)));
    } catch (err: any) {
      console.error('Query error:', err);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id
            ? {
                ...t,
                error: err.message || 'Failed to retrieve guidelines. Ensure the backend server is running.',
                loading: false,
              }
            : t
        )
      );
    }
  };

  const handleSelectGoldQuestion = (questionText: string) => {
    setQuery(questionText);
    setActiveTab('search');
  };

  return (
    <div className="app-container">
      <Header health={health} activeTab={activeTab} setActiveTab={setActiveTab} />

      {activeTab === 'search' && (
        <div className="chat-layout">
          <div className="chat-scroll-area" ref={scrollRef}>
            <div className="chat-scroll-inner">
                  {turns.length === 0 && (
                <div className="chat-empty-state">
                  {(() => {
                    const hour = new Date().getHours();
                    const hello = hour < 12 ? 'Good morning' : hour < 17 ? 'Good afternoon' : 'Good evening';
                    return `${hello}! 👋 I'm your H. pylori information assistant. How can I help you today?`;
                  })()}
                </div>
              )}

              {turns.map((turn) => (
                <div key={turn.id} className="chat-turn">
                  <div className="chat-user-message">
                    <User size={15} />
                    <span>{turn.query}</span>
                  </div>

                  {turn.loading && (
                    <div className="card">
                      <div
                        className="card-body"
                        style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: '#6B7280' }}
                      >
                        <span className="spinner"></span>
                        Retrieving &amp; reranking guideline evidence...
                      </div>
                    </div>
                  )}

                  {turn.error && (
                    <div className="card" style={{ borderColor: '#C62828', background: '#ffebee' }}>
                      <div
                        className="card-body"
                        style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', color: '#C62828' }}
                      >
                        <AlertCircle size={20} />
                        <div>
                          <strong>Query Error:</strong> {turn.error}
                        </div>
                      </div>
                    </div>
                  )}

                  {turn.response && (
                    <>
                      <RecommendationBox response={turn.response} onSelectFollowup={(q) => setQuery(q)} />
                      {turn.response.answer_status !== 'greeting' && turn.response.answer_status !== 'casual' && (
                        <PipelineTrace response={turn.response} />
                      )}
                      {turn.response.answer_status === 'answered' && (
                        <>
                          <GroundedEvidence citations={turn.response.citations} />
                          <RerankedDocs documents={turn.response.reranked_documents} />
                        </>
                      )}
                    </>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div className="chat-input-bar">
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
          </div>
        </div>
      )}

      {activeTab === 'gold' && (
        <main className="main-content">
          <GoldWorkbench onSelectQuestion={handleSelectGoldQuestion} />
        </main>
      )}
    </div>
  );
};

export default App;
