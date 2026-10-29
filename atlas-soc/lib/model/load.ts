import { readFileSync } from "node:fs";
import path from "node:path";
import { load as parseYaml } from "js-yaml";
import { modelSchema, type ArchitectureModel } from "./schema";

const MODEL_PATH = path.join(process.cwd(), "architecture", "soc.yaml");

let cached: ArchitectureModel | null = null;

/**
 * Parses and validates architecture/soc.yaml once per server process.
 *
 * Failure is loud on purpose: a model that does not satisfy the schema means
 * every downstream surface would render invented or missing data, which the
 * product principles forbid. scripts/validate_model.py gates this in CI; this
 * is the runtime backstop.
 */
export function loadModel(): ArchitectureModel {
  if (cached) return cached;
  const raw = parseYaml(readFileSync(MODEL_PATH, "utf8"));
  const parsed = modelSchema.safeParse(raw);
  if (!parsed.success) {
    const issues = parsed.error.issues
      .slice(0, 20)
      .map((issue) => `  ${issue.path.join(".") || "<root>"}: ${issue.message}`)
      .join("\n");
    throw new Error(
      `architecture/soc.yaml failed validation (${parsed.error.issues.length} issue(s)):\n${issues}\n` +
        "Run `python3 scripts/validate_model.py` for the canonical report.",
    );
  }
  cached = parsed.data;
  return cached;
}
