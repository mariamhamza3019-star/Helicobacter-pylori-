import React, { useEffect, useState } from 'react';
import { BookOpen, Play } from 'lucide-react';
import { GoldQuestion } from '../types';

interface GoldWorkbenchProps {
  onSelectQuestion: (questionText: string) => void;
}

export const GoldWorkbench: React.FC<GoldWorkbenchProps> = ({ onSelectQuestion }) => {
  const [questions, setQuestions] = useState<GoldQuestion[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>('');

  useEffect(() => {
    fetch('/api/gold-questions')
      .then((res) => res.json())
      .then((data) => {
        setQuestions(data.questions || []);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load gold questions:', err);
        setLoading(false);
      });
  }, []);

  const filtered = questions.filter((g) =>
    g.q.toLowerCase().includes(searchTerm.toLowerCase()) ||
    g.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
    g.expect_sections.some((s) => s.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  return (
    <div className="card">
      <div className="card-header">
        <h3>
          <BookOpen size={18} style={{ color: '#1E3A5F' }} />
          ACG Gold Evaluation Benchmark ({questions.length} Questions)
        </h3>
        <span style={{ fontSize: '0.78rem', color: '#6B7280', fontWeight: 600 }}>
          Standard Clinician Validation Set
        </span>
      </div>
      <div className="card-body">
        <div style={{ marginBottom: '1rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <input
            type="text"
            className="query-input"
            style={{ padding: '0.5rem 0.8rem', fontSize: '0.86rem' }}
            placeholder="Filter gold questions by ID, text, or expected section..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '2rem', color: '#6B7280' }}>
            Loading gold questions...
          </div>
        ) : (
          <div className="gold-grid">
            {filtered.map((g) => (
              <div
                key={g.id}
                className="gold-item-card"
                onClick={() => onSelectQuestion(g.q)}
              >
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="gold-id-tag">{g.id}</span>
                    <button
                      className="btn-primary"
                      style={{ padding: '0.2rem 0.55rem', fontSize: '0.72rem' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectQuestion(g.q);
                      }}
                    >
                      <Play size={11} /> Test Query
                    </button>
                  </div>
                  <div className="gold-question-text">{g.q}</div>
                </div>

                <div className="gold-expected-sections">
                  <strong>Expected Section(s):</strong> {g.expect_sections.join(', ')}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
