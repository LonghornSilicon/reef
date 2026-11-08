"use client";

import { Icon } from "@/components/icons";
import type { ViewSnapshot } from "@/lib/model/view";
import { MATURITY_COLOR, MATURITY_LABEL, MATURITY_ORDER, type Lens } from "./lens";

type Props = {
  snapshot: ViewSnapshot;
  lens: Lens;
  onLens(lens: Lens): void;
};

/**
 * Lens switching and the legend that each lens requires.
 *
 * Protocol and maturity colours encode named engineering states, so the legend
 * is not optional decoration — it is the control itself.
 */
export function LensControls({ snapshot, lens, onLens }: Props) {
  const protocolIds = Object.keys(snapshot.protocols).filter((id) => id !== "containment");
  const sequence = lens.kind === "sequence" ? snapshot.sequences.find((item) => item.id === lens.sequenceId) : undefined;

  return (
    <div className="lens-bar">
      <div className="lens-switch" role="group" aria-label="Diagram lens">
        <button type="button" className={lens.kind === "structure" ? "active" : ""} onClick={() => onLens({ kind: "structure" })}>
          Structure
        </button>
        <button
          type="button"
          className={lens.kind === "protocol" ? "active" : ""}
          onClick={() => onLens({ kind: "protocol", protocol: lens.kind === "protocol" ? lens.protocol : protocolIds[0] ?? "" })}
        >
          Protocol
        </button>
        <button type="button" className={lens.kind === "readiness" ? "active" : ""} onClick={() => onLens({ kind: "readiness" })}>
          Readiness
        </button>
        <button
          type="button"
          className={lens.kind === "sequence" ? "active" : ""}
          onClick={() =>
            onLens({
              kind: "sequence",
              sequenceId: lens.kind === "sequence" ? lens.sequenceId : snapshot.sequences[0]?.id ?? "",
              step: 0,
            })
          }
        >
          Sequence
        </button>
      </div>

      {lens.kind === "protocol" && (
        <div className="lens-detail protocol-legend" role="group" aria-label="Protocol">
          {protocolIds.map((id) => {
            const protocol = snapshot.protocols[id];
            const active = lens.protocol === id;
            return (
              <button
                key={id}
                type="button"
                className={active ? "active" : ""}
                aria-pressed={active}
                title={protocol.purpose}
                onClick={() => onLens({ kind: "protocol", protocol: id })}
              >
                <span style={{ background: protocol.color }} aria-hidden="true" />
                {protocol.abbreviation}
              </button>
            );
          })}
        </div>
      )}

      {lens.kind === "readiness" && (
        <div className="lens-detail readiness-legend">
          {MATURITY_ORDER.map((maturity) => (
            <span key={maturity} className={snapshot.counts.byMaturity[maturity] ? "" : "absent"}>
              <span style={{ background: MATURITY_COLOR[maturity] }} aria-hidden="true" />
              {MATURITY_LABEL[maturity]}
              <small>{snapshot.counts.byMaturity[maturity] ?? 0}</small>
            </span>
          ))}
        </div>
      )}

      {lens.kind === "sequence" && (
        <div className="lens-detail sequence-control">
          <label>
            <span className="sr-only">Sequence</span>
            <select
              value={lens.sequenceId}
              onChange={(event) => onLens({ kind: "sequence", sequenceId: event.target.value, step: 0 })}
            >
              {snapshot.sequences.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.kind === "boot" ? "Behaviour" : "Instruction"} · {item.name}
                </option>
              ))}
            </select>
          </label>
          {sequence && (
            <div className="sequence-stepper">
              <button
                type="button"
                aria-label="Previous step"
                disabled={lens.step === 0}
                onClick={() => onLens({ ...lens, step: Math.max(0, lens.step - 1) })}
              >
                <Icon name="chevron" style={{ transform: "rotate(180deg)" }} />
              </button>
              <output>
                {Math.min(lens.step, sequence.steps.length - 1) + 1} / {sequence.steps.length}
              </output>
              <button
                type="button"
                aria-label="Next step"
                disabled={lens.step >= sequence.steps.length - 1}
                onClick={() => onLens({ ...lens, step: Math.min(sequence.steps.length - 1, lens.step + 1) })}
              >
                <Icon name="chevron" />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function SequenceStepDetail({ snapshot, lens }: { snapshot: ViewSnapshot; lens: Lens }) {
  if (lens.kind !== "sequence") return null;
  const sequence = snapshot.sequences.find((item) => item.id === lens.sequenceId);
  if (!sequence) return null;
  const index = Math.min(lens.step, sequence.steps.length - 1);
  const step = sequence.steps[index];
  if (!step) return null;
  return (
    <div className="sequence-detail">
      <span className="sequence-eyebrow">
        Step {index + 1} of {sequence.steps.length}
        {step.parallel && <em> · concurrent</em>}
        {step.branch && <em> · alternative path</em>}
      </span>
      <strong>{step.title}</strong>
      <p>{step.detail}</p>
    </div>
  );
}
