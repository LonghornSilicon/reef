export default function Loading() {
  return (
    <div className="page" aria-busy="true" aria-label="Loading process view">
      <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1fr) 336px", gridTemplateRows: "auto minmax(0, 1fr)" }}>
        <div className="float pad stack" style={{ gridColumn: "1 / -1" }}><div className="sk sk-line" style={{ width: 240, height: 20 }} /><div className="sk sk-block" /></div>
        <div className="float pad"><div className="pipe">{[1, 1, 4, 1, 1, 1, 1].map((n, i) => (
          <div key={i} className={`stage${n > 1 ? " q" : ""}`}><div className="sk sk-line" style={{ width: "60%" }} /><div className="stage-box">{Array.from({ length: n }, (_, j) => <div key={j} className="sk" style={{ height: 44 }} />)}</div></div>
        ))}</div></div>
        <div className="float pad stack">{Array.from({ length: 5 }, (_, i) => <div key={i} className="sk sk-line" />)}</div>
      </div>
    </div>
  );
}
