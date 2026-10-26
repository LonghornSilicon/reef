import Link from "next/link";
import type { BranchContext } from "@/lib/server/context";
import { Empty, Note } from "./ui";

/** What a view shows when its branch has no usable spec. */
export function ContextState({ ctx }: { ctx: Exclude<BranchContext, { kind: "ok" }> }) {
  if (ctx.kind === "no-spec") {
    return (
      <div className="page" style={{ display: "grid", placeItems: "center" }}>
        <Empty icon="{}" title={`No spec on ${ctx.branch.name}`} action={<Link className="btn primary" href="?">Open main</Link>}>
          Atlas reads <code>architecture/soc.yaml</code> from the branch. Add it (copy it from main to start), push, and this view fills in.
        </Empty>
      </div>
    );
  }
  return (
    <div className="page" style={{ display: "grid", placeItems: "center" }}>
      <div style={{ maxWidth: 640, width: "100%" }}>
        <Note kind="err" action={<Link className="btn sm" href="?">Open main</Link>}>
          <b>The spec on {ctx.branch.name} does not validate.</b> Nothing is drawn until these are fixed:
          <ul>{ctx.issues.slice(0, 20).map((i) => <li key={i}>{i}</li>)}</ul>
        </Note>
      </div>
    </div>
  );
}
