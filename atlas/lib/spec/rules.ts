/**
 * Timing-rule expressions (plan.md §M1.1): signal identifiers, `! && || == !=`,
 * integer literals, parentheses and `$past(expr)`. Parsed once, evaluated per cycle.
 */

export type Expr =
  | { t: "sig"; name: string }
  | { t: "num"; value: number }
  | { t: "not"; e: Expr }
  | { t: "past"; e: Expr }
  | { t: "bin"; op: "&&" | "||" | "==" | "!="; a: Expr; b: Expr };

export class RuleSyntaxError extends Error {}

const TOKEN = /\s*(\$past|&&|\|\||==|!=|!|\(|\)|[0-9]+|[A-Za-z_][A-Za-z0-9_.[\]]*)/y;

function tokenize(src: string): string[] {
  const out: string[] = [];
  TOKEN.lastIndex = 0;
  while (TOKEN.lastIndex < src.length) {
    const start = TOKEN.lastIndex;
    const m = TOKEN.exec(src);
    if (!m) {
      if (src.slice(start).trim() === "") break;
      throw new RuleSyntaxError(`Unexpected character at ${start} in "${src}"`);
    }
    out.push(m[1]);
  }
  return out;
}

export function parseExpr(src: string): Expr {
  const toks = tokenize(src);
  let i = 0;
  const peek = () => toks[i];
  const eat = (t?: string) => {
    const tok = toks[i];
    if (t !== undefined && tok !== t) throw new RuleSyntaxError(`Expected "${t}" but found "${tok ?? "end"}" in "${src}"`);
    i += 1;
    return tok;
  };
  const primary = (): Expr => {
    const tok = peek();
    if (tok === undefined) throw new RuleSyntaxError(`Unexpected end of "${src}"`);
    if (tok === "!") { eat(); return { t: "not", e: primary() }; }
    if (tok === "(") { eat(); const e = or(); eat(")"); return e; }
    if (tok === "$past") { eat(); eat("("); const e = or(); eat(")"); return { t: "past", e }; }
    if (/^[0-9]+$/.test(tok)) { eat(); return { t: "num", value: Number(tok) }; }
    if (/^[A-Za-z_]/.test(tok)) { eat(); return { t: "sig", name: tok }; }
    throw new RuleSyntaxError(`Unexpected "${tok}" in "${src}"`);
  };
  const cmp = (): Expr => {
    let a = primary();
    while (peek() === "==" || peek() === "!=") { const op = eat() as "==" | "!="; a = { t: "bin", op, a, b: primary() }; }
    return a;
  };
  const and = (): Expr => {
    let a = cmp();
    while (peek() === "&&") { eat(); a = { t: "bin", op: "&&", a, b: cmp() }; }
    return a;
  };
  const or = (): Expr => {
    let a = and();
    while (peek() === "||") { eat(); a = { t: "bin", op: "||", a, b: and() }; }
    return a;
  };
  const e = or();
  if (i !== toks.length) throw new RuleSyntaxError(`Unexpected "${toks[i]}" in "${src}"`);
  return e;
}

/** Every signal identifier an expression reads. */
export function identifiers(e: Expr, out = new Set<string>()): Set<string> {
  if (e.t === "sig") out.add(e.name.split("[")[0]);
  else if (e.t === "not" || e.t === "past") identifiers(e.e, out);
  else if (e.t === "bin") { identifiers(e.a, out); identifiers(e.b, out); }
  return out;
}

/** Evaluate at `cycle`; `read(name, cycle)` returns the signal's value (missing → 0). */
export function evaluate(e: Expr, cycle: number, read: (name: string, cycle: number) => number): number {
  switch (e.t) {
    case "num": return e.value;
    case "sig": return read(e.name, cycle);
    case "not": return evaluate(e.e, cycle, read) ? 0 : 1;
    case "past": return evaluate(e.e, cycle - 1, read);
    case "bin": {
      const a = evaluate(e.a, cycle, read);
      if (e.op === "&&") return a && evaluate(e.b, cycle, read) ? 1 : 0;
      if (e.op === "||") return a || evaluate(e.b, cycle, read) ? 1 : 0;
      const b = evaluate(e.b, cycle, read);
      return (e.op === "==" ? a === b : a !== b) ? 1 : 0;
    }
  }
}
