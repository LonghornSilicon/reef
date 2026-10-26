/** Mirrors the canvas: outline, toolbar, a few block shapes and the legend. */
export default function Loading() {
  return (
    <div className="viewport" aria-busy="true" aria-label="Loading canvas">
      <div className="outline float" style={{ padding: 12, gap: 8 }}>
        <div className="sk sk-line" style={{ width: "40%" }} />
        <div className="sk sk-block" />
        {Array.from({ length: 12 }, (_, i) => <div key={i} className="sk sk-line" style={{ width: `${60 + ((i * 17) % 35)}%`, marginLeft: (i % 3) * 12 }} />)}
      </div>
      <div className="c-tools float" style={{ left: 288, top: 72, width: 320, height: 40 }}><div className="sk" style={{ flex: 1, height: 24 }} /></div>
      {[[340, 160, 300, 140], [680, 160, 420, 300], [340, 340, 300, 140], [1140, 160, 220, 180], [520, 520, 300, 120]].map(([x, y, w, h], i) => (
        <div key={i} className="sk" style={{ position: "absolute", left: x, top: y, width: w, height: h, borderRadius: 8 }} />
      ))}
      <div className="legend float" style={{ width: 560, height: 40 }}><div className="sk" style={{ flex: 1, height: 16 }} /></div>
    </div>
  );
}
