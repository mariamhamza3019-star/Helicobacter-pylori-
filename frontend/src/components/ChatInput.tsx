import React, { useState, useRef, useEffect } from 'react';
import { Send, SlidersHorizontal, Sparkles } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  loading: boolean;
  topK: number;
  setTopK: (k: number) => void;
  pipeline: string;
  setPipeline: (p: string) => void;
  relevanceThreshold: number;
  setRelevanceThreshold: (t: number) => void;
  hasMessages: boolean;
  onSelectQuick: (q: string) => void;
}

const IN_CHAT_QUICK_CHIPS = [
  'First-line BQT duration?',
  'PCAB dual therapy evidence?',
  'When to use rifabutin triple?',
  'Confirming cure with UBT vs Serology'
];

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  loading,
  topK,
  setTopK,
  pipeline,
  setPipeline,
  relevanceThreshold,
  setRelevanceThreshold,
  hasMessages,
  onSelectQuick
}) => {
  const [text, setText] = useState<string>('');
  const [showSettings, setShowSettings] = useState<boolean>(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  }, [text]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = text.trim();
    if (!trimmed || loading) return;
    onSend(trimmed);
    setText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="chat-input-sticky-footer">
      <div className="chat-input-inner">
        {/* In-chat quick question chips if user is already chatting */}
        {hasMessages && (
          <div className="in-chat-chips-row">
            <span className="chips-hint">
              <Sparkles size={12} /> Suggestions:
            </span>
            {IN_CHAT_QUICK_CHIPS.map((chip, idx) => (
              <button
                key={idx}
                className="in-chat-chip"
                onClick={() => onSelectQuick(chip)}
                disabled={loading}
              >
                {chip}
              </button>
            ))}
          </div>
        )}

        {/* Input box */}
        <form onSubmit={handleSubmit} className="input-form-box">
          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder="Ask a clinical question about H. pylori treatment, diagnosis, salvage, or resistance..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading}
            rows={1}
          />

          <div className="input-actions-bar">
            <button
              type="button"
              className={`btn-settings-toggle ${showSettings ? 'is-active' : ''}`}
              title="Pipeline settings"
              onClick={() => setShowSettings(!showSettings)}
            >
              <SlidersHorizontal size={16} />
              <span className="btn-label-text">Settings</span>
            </button>

            <button
              type="submit"
              className="btn-send-message"
              disabled={loading || !text.trim()}
              title="Send message (Enter)"
            >
              {loading ? (
                <span className="send-spinner"></span>
              ) : (
                <Send size={16} />
              )}
            </button>
          </div>
        </form>

        {/* Collapsible settings drawer */}
        {showSettings && (
          <div className="input-settings-panel">
            <div className="setting-control">
              <label>Retrieval Pipeline:</label>
              <select
                value={pipeline}
                onChange={(e) => setPipeline(e.target.value)}
                disabled={loading}
              >
                <option value="rrf_rerank">RRF Hybrid + MedCPT Cross-Encoder (Shipping Stack)</option>
                <option value="dense_rerank">Dense FAISS + MedCPT Rerank</option>
                <option value="rrf">RRF Hybrid (No Reranker)</option>
                <option value="minmax">MinMax 70/30 Hybrid</option>
                <option value="bm25">BM25 Lexical Only</option>
              </select>
            </div>

            <div className="setting-control">
              <label>Top K Chunks:</label>
              <select
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={loading}
              >
                <option value={3}>Top 3</option>
                <option value={5}>Top 5 (Recommended)</option>
                <option value={8}>Top 8</option>
                <option value={10}>Top 10</option>
              </select>
            </div>

            <div className="setting-control">
              <label>Relevance Threshold:</label>
              <input
                type="number"
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

        <div className="disclaimer-text">
          ACG Clinical Guideline 2024 · Grounded with BM25 + FAISS + MedCPT Reranking · For clinical decision support only
        </div>
      </div>
    </div>
  );
};
