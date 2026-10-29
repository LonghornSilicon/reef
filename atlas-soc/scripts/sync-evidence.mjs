/**
 * Stages the assets the web application serves directly.
 *
 * Replaces scripts/sync-legacy.mjs, which mirrored the whole static viewer into
 * public/legacy so the Next app could iframe it. The viewer is now native and
 * the font is self-hosted by next/font, so the only things left to copy are the
 * binary evidence files that architecture/soc.yaml actually references.
 *
 * The evidence list is derived from the model, so new references are picked up
 * without touching this script.
 */
import { cp, mkdir, readFile, rm } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { load as parseYaml } from "js-yaml";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const publicDir = path.join(root, "public");

const model = parseYaml(await readFile(path.join(root, "architecture", "soc.yaml"), "utf8"));

const referenced = new Set();
const collect = (documents) => {
  for (const document of documents ?? []) {
    if (!document?.path) continue;
    // docs/*.md are rendered as wiki pages, not served as files.
    if (/^docs\/.+\.md$/.test(document.path)) continue;
    referenced.add(document.path);
  }
};
for (const block of model.blocks ?? []) collect(block.documents);
for (const behavior of model.behaviors ?? []) collect(behavior.documents);

const evidenceDir = path.join(publicDir, "evidence");
await rm(evidenceDir, { recursive: true, force: true });
await mkdir(evidenceDir, { recursive: true });

const missing = [];
let copied = 0;
for (const relative of [...referenced].sort()) {
  const source = path.join(root, relative);
  if (!existsSync(source)) {
    missing.push(relative);
    continue;
  }
  const destination = path.join(evidenceDir, relative);
  await mkdir(path.dirname(destination), { recursive: true });
  await cp(source, destination);
  copied += 1;
}

const logo = path.join(root, "logo.png");
if (existsSync(logo)) {
  await mkdir(publicDir, { recursive: true });
  await cp(logo, path.join(publicDir, "logo.png"));
}

console.log(`sync-evidence: staged ${copied} evidence file(s) into public/evidence`);
if (missing.length) {
  console.warn(`sync-evidence: ${missing.length} referenced file(s) missing from the repository:`);
  for (const relative of missing) console.warn(`  ${relative}`);
}
