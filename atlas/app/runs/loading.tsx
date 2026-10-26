export default function Loading() {
  return (
    <div className="page" aria-busy="true" aria-label="Loading runs">
      <div className="cols" style={{ gridTemplateColumns: "320px minmax(0, 1fr)" }}>
        <div className="float pad stack">{Array.from({ length: 6 }, (_, i) => <div key={i} className="sk" style={{ height: 52, borderRadius: 8 }} />)}</div>
        <div className="stack">
          <div className="float pad stack"><div className="sk sk-line" style={{ width: "30%", height: 20 }} /><div className="sk sk-line" style={{ width: "70%" }} /></div>
          <div className="float pad stack">{Array.from({ length: 3 }, (_, i) => <div key={i} className="sk sk-line" />)}</div>
          <div className="float pad stack" style={{ flex: 1 }}>{Array.from({ length: 10 }, (_, i) => <div key={i} className="sk sk-line" style={{ width: `${50 + ((i * 29) % 45)}%` }} />)}</div>
        </div>
      </div>
    </div>
  );
}
