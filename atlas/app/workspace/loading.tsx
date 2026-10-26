export default function Loading() {
  return (
    <div className="page" aria-busy="true" aria-label="Loading workspace">
      <div className="ws-bar float"><div className="sk" style={{ width: 72, height: 32 }} /><div className="sk grow" style={{ height: 8 }} /></div>
      <div className="ws">
        {Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="ws-pane float pad stack">
            <div className="sk sk-line" style={{ width: 120 }} />
            {Array.from({ length: 5 }, (_, j) => <div key={j} className="sk sk-line" style={{ width: `${50 + ((i * 7 + j * 13) % 45)}%` }} />)}
          </div>
        ))}
      </div>
    </div>
  );
}
