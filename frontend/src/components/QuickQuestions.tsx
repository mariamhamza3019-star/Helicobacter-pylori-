import React from 'react';
import { Pill, AlertOctagon, ShieldAlert, CheckCircle2, Sparkles, ArrowRight } from 'lucide-react';

interface QuickQuestionsProps {
  onSelect: (questionText: string) => void;
  disabled?: boolean;
}

interface QuestionCardItem {
  title: string;
  query: string;
  description: string;
  icon: React.ReactNode;
  tag: string;
}

const QUICK_QUESTIONS: QuestionCardItem[] = [
  {
    title: 'First-line H. pylori treatment?',
    query: 'What is the preferred first-line treatment for treatment-naive patients when antibiotic susceptibility is unknown?',
    description: 'Bismuth quadruple therapy (BQT), PCAB dual therapy, and duration guidelines.',
    icon: <Pill size={20} color="#2F6690" />,
    tag: 'Treatment-Naive'
  },
  {
    title: 'Salvage treatment after failure?',
    query: 'Which salvage regimens are recommended for treatment-experienced patients with persistent H. pylori infection?',
    description: 'Management following first-line failure, rifabutin triple, and optimized BQT.',
    icon: <AlertOctagon size={20} color="#D97706" />,
    tag: 'Treatment-Experienced'
  },
  {
    title: 'Treatment for penicillin allergy?',
    query: 'How should H. pylori be managed in patients with a confirmed penicillin allergy?',
    description: 'Safe alternative regimens without amoxicillin and AST recommendations.',
    icon: <ShieldAlert size={20} color="#C62828" />,
    tag: 'Allergy Management'
  },
  {
    title: 'When to confirm eradication?',
    query: 'How long after completing therapy should post-treatment testing to confirm H. pylori eradication be performed, and with which tests?',
    description: 'Timing of test-of-cure, UBT, stool antigen, and PPI washout requirements.',
    icon: <CheckCircle2 size={20} color="#2E7D32" />,
    tag: 'Test of Cure'
  }
];

export const QuickQuestions: React.FC<QuickQuestionsProps> = ({ onSelect, disabled }) => {
  return (
    <div className="quick-questions-container">
      <div className="quick-questions-header">
        <div className="quick-badge">
          <Sparkles size={14} /> Quick Questions
        </div>
        <p className="quick-subtitle">Select a common clinical question or type your own below</p>
      </div>

      <div className="quick-cards-grid">
        {QUICK_QUESTIONS.map((item, idx) => (
          <button
            key={idx}
            className="quick-card-btn"
            onClick={() => onSelect(item.query)}
            disabled={disabled}
          >
            <div className="quick-card-top">
              <div className="quick-card-icon">{item.icon}</div>
              <span className="quick-card-tag">{item.tag}</span>
            </div>
            <div className="quick-card-title">{item.title}</div>
            <div className="quick-card-desc">{item.description}</div>
            <div className="quick-card-footer">
              <span>Ask Guideline</span>
              <ArrowRight size={14} className="arrow-hover-icon" />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};
