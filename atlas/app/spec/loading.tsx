export default function Loading() {
  return (
    <div className="page" aria-busy="true" aria-label="Loading doc graph">
      <div className="cols" style={{ gridTemplateColumns: "minmax(0, 1fr) 384px" }}>
        <div className="float pad" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", columnGap: 96 }}>
          {[14, 3, 6, 4].map((n, c) => (
            <div key={c} className="stack">
              <div className="sk sk-line" style={{ width: "50%" }} />
              {Array.from({ length: n }, (_, i) => <div key={i} className="sk sk-block" />)}
            </div>
          ))}
        </div>
        <div className="float pad stack">{Array.from({ length: 6 }, (_, i) => <div key={i} className="sk sk-line" />)}</div>
      </div>
    </div>
  );
}
