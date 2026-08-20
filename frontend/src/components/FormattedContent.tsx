import React from 'react';

interface FormattedContentProps {
  content: string;
  className?: string;
}

export const FormattedContent: React.FC<FormattedContentProps> = ({ content, className = '' }) => {
  if (!content) return null;

  // Split content into blocks by double newlines or table boundary
  const lines = content.split('\n');
  const blocks: React.ReactNode[] = [];
  let tableLines: string[] = [];
  let inTable = false;
  let textParagraphs: string[] = [];

  const flushText = (keyPrefix: string) => {
    if (textParagraphs.length > 0) {
      const text = textParagraphs.join('\n').trim();
      if (text) {
        blocks.push(
          <div key={`${keyPrefix}-${blocks.length}`} className="formatted-paragraph">
            {formatInlineText(text)}
          </div>
        );
      }
      textParagraphs = [];
    }
  };

  const flushTable = (keyPrefix: string) => {
    if (tableLines.length > 0) {
      blocks.push(
        <div key={`${keyPrefix}-table-${blocks.length}`} className="table-responsive-wrapper">
          {renderMarkdownTable(tableLines)}
        </div>
      );
      tableLines = [];
      inTable = false;
    }
  };

  lines.forEach((line) => {
    const trimmed = line.trim();
    const isTableLine = trimmed.startsWith('|') && trimmed.endsWith('|');

    if (isTableLine) {
      flushText('pre-table');
      inTable = true;
      tableLines.push(trimmed);
    } else {
      if (inTable) {
        flushTable('post-table');
      }
      textParagraphs.push(line);
    }
  });

  flushText('final-text');
  flushTable('final-table');

  return <div className={`formatted-content-container ${className}`}>{blocks}</div>;
};

// Helper to render markdown table lines into clean HTML table
function renderMarkdownTable(tableLines: string[]): React.ReactNode {
  if (tableLines.length === 0) return null;

  const rows = tableLines.map((row) =>
    row
      .split('|')
      .slice(1, -1) // remove leading and trailing empty items from '| cell | cell |'
      .map((cell) => cell.trim())
  );

  // Find if line 1 (index 1) is separator (|---|---|)
  let headerRow: string[] = [];
  let bodyRows: string[][] = [];

  if (rows.length >= 2 && rows[1].every((cell) => /^:?-+:?$/.test(cell))) {
    headerRow = rows[0];
    bodyRows = rows.slice(2);
  } else {
    bodyRows = rows;
  }

  return (
    <table className="medical-table">
      {headerRow.length > 0 && (
        <thead>
          <tr>
            {headerRow.map((cell, idx) => (
              <th key={idx}>{formatInlineText(cell) || '\u00A0'}</th>
            ))}
          </tr>
        </thead>
      )}
      <tbody>
        {bodyRows.map((row, rowIdx) => {
          // Filter out rows that are entirely empty
          const hasContent = row.some((c) => c.length > 0);
          if (!hasContent) return null;

          return (
            <tr key={rowIdx}>
              {row.map((cell, cellIdx) => (
                <td key={cellIdx}>{formatInlineText(cell) || '\u00A0'}</td>
              ))}
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// Helper for inline text formatting (bold, code, lists)
function formatInlineText(text: string): React.ReactNode {
  if (!text) return null;

  // Split lines if paragraph contains list items
  const lines = text.split('\n');
  return lines.map((line, lIdx) => {
    // Check for bold **text**
    const parts = line.split(/(\*\*.*?\*\*|`.*?`)/g);
    const formattedLine = parts.map((part, pIdx) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={pIdx}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('`') && part.endsWith('`')) {
        return <code key={pIdx} className="inline-code">{part.slice(1, -1)}</code>;
      }
      return part;
    });

    return (
      <React.Fragment key={lIdx}>
        {formattedLine}
        {lIdx < lines.length - 1 && <br />}
      </React.Fragment>
    );
  });
}
