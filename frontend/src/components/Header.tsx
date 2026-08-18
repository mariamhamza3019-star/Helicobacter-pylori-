import React from 'react';
import { Stethoscope, Sparkles, BookOpen } from 'lucide-react';
import { HealthStatus } from '../types';

interface HeaderProps {
  health: HealthStatus | null;
  activeTab: 'search' | 'gold';
  setActiveTab: (tab: 'search' | 'gold') => void;
}

export const Header: React.FC<HeaderProps> = ({ health, activeTab, setActiveTab }) => {
  return (
    <header className="header-wrapper">
      <div className="header-content">
        <div className="header-brand">
          <div className="header-icon-badge">
            <Stethoscope size={24} />
          </div>
          <div className="header-title-block">
            <h1>H. pylori Clinical Decision Support</h1>
            <p>ACG 2024 Clinical Guideline · Evidence-Grounded Hybrid RAG</p>
          </div>
        </div>

        <div className="header-actions">
          <div className="nav-tabs">
            <button
              className={`nav-tab-btn ${activeTab === 'search' ? 'active' : ''}`}
              onClick={() => setActiveTab('search')}
            >
              <Sparkles size={15} />
              Clinical Query
            </button>
            <button
              className={`nav-tab-btn ${activeTab === 'gold' ? 'active' : ''}`}
              onClick={() => setActiveTab('gold')}
            >
              <BookOpen size={15} />
              Gold Benchmark
            </button>
          </div>

          <div className="status-badge">
            <span className="status-dot"></span>
            {health ? (
              <span>{health.num_chunks} Chunks Loaded</span>
            ) : (
              <span>Connecting...</span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
