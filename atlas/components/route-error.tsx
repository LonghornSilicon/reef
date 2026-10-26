"use client";

import { useEffect } from "react";
import { Note } from "./ui";

/** Inline, non-disruptive error for a view; the shell stays usable. */
export function RouteError({ error, reset, what }: { error: Error & { digest?: string }; reset: () => void; what: string }) {
  useEffect(() => { console.error(error); }, [error]);
  return (
    <div className="page" style={{ display: "grid", placeItems: "center" }}>
      <div style={{ maxWidth: 560, width: "100%" }}>
        <Note kind="err" action={<button type="button" className="btn sm" onClick={reset}>Try again</button>}>
          <b>{what} could not load.</b> {error.digest ? `Server error ${error.digest}.` : error.message}
        </Note>
      </div>
    </div>
  );
}
