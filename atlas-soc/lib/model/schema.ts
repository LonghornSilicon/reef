import { z } from "zod";

/**
 * Mirrors architecture/schema.json and the invariants enforced by
 * scripts/validate_model.py. The Python validator remains the canonical gate in
 * CI; this schema is what lets TypeScript read the same model at runtime.
 */

export const MATURITY = ["proposal", "specified", "in-progress", "implemented", "verified"] as const;
export const SIDES = ["left", "right", "top", "bottom"] as const;

export type Maturity = (typeof MATURITY)[number];
export type Side = (typeof SIDES)[number];

const documentSchema = z.object({
  title: z.string(),
  path: z.string().optional(),
  url: z.string().optional(),
  anchor: z.string().optional(),
  page: z.number().optional(),
});

const issueSchema = z.object({
  url: z.string(),
  title: z.string().optional(),
});

const rtlSchema = z.object({
  repository: z.string(),
  revision: z.string(),
  path: z.string(),
  symbol: z.string(),
  url: z.string(),
});

const blockSchema = z.object({
  id: z.string(),
  name: z.string(),
  kind: z.string(),
  summary: z.string(),
  parent: z.string().nullable(),
  owner: z.string(),
  maturity: z.enum(MATURITY),
  rank: z.number().int().nonnegative(),
  order: z.number().int().nonnegative(),
  responsibilities: z.array(z.string()).min(1),
  documents: z.array(documentSchema),
  issues: z.array(issueSchema),
  rtl: z.array(rtlSchema),
});

const protocolSchema = z.object({
  name: z.string(),
  color: z.string(),
  abbreviation: z.string(),
  purpose: z.string(),
  roles: z.array(z.string()),
});

const portSchema = z.object({
  label: z.string(),
  protocol: z.string(),
  side: z.enum(SIDES),
  offset: z.number().min(0).max(1),
});

const rectSchema = z.object({
  x: z.number(),
  y: z.number(),
  width: z.number().positive(),
  height: z.number().positive(),
});

const layoutSchema = z.object({
  canvas: z.object({ width: z.number().positive(), height: z.number().positive() }),
  nodes: z.record(z.string(), rectSchema),
});

const connectionSchema = z.object({
  id: z.string(),
  from: z.string(),
  from_port: z.string(),
  to: z.string(),
  to_port: z.string(),
  protocol: z.string(),
  label: z.string(),
  bidirectional: z.boolean(),
});

const flowStepSchema = z.object({
  title: z.string(),
  detail: z.string(),
  blocks: z.array(z.string()),
  protocol: z.string(),
});

const instructionFlowSchema = z.object({
  id: z.string(),
  name: z.string(),
  summary: z.string(),
  steps: z.array(flowStepSchema).min(1),
});

const behaviorStepSchema = z.object({
  id: z.string(),
  title: z.string(),
  detail: z.string(),
  blocks: z.array(z.string()),
  parallel: z.boolean().optional(),
  branch: z.boolean().optional(),
});

const behaviorSchema = z.object({
  id: z.string(),
  name: z.string(),
  summary: z.string(),
  documents: z.array(documentSchema).optional(),
  steps: z.array(behaviorStepSchema).min(1),
});

export const modelSchema = z.object({
  schema_version: z.literal(1),
  product: z.object({
    id: z.string(),
    name: z.string(),
    description: z.string(),
    default_focus: z.string(),
  }),
  owners: z.array(z.object({ id: z.string(), name: z.string() })),
  protocols: z.record(z.string(), protocolSchema),
  instruction_flows: z.array(instructionFlowSchema),
  layouts: z.record(z.string(), layoutSchema),
  ports: z.record(z.string(), z.record(z.string(), portSchema)),
  blocks: z.array(blockSchema),
  connections: z.array(connectionSchema),
  behaviors: z.array(behaviorSchema),
});

export type ArchitectureModel = z.infer<typeof modelSchema>;
export type Block = z.infer<typeof blockSchema>;
export type Protocol = z.infer<typeof protocolSchema>;
export type Port = z.infer<typeof portSchema>;
export type Connection = z.infer<typeof connectionSchema>;
export type Layout = z.infer<typeof layoutSchema>;
export type Rect = z.infer<typeof rectSchema>;
export type InstructionFlow = z.infer<typeof instructionFlowSchema>;
export type Behavior = z.infer<typeof behaviorSchema>;
export type BehaviorStep = z.infer<typeof behaviorStepSchema>;
export type EvidenceDocument = z.infer<typeof documentSchema>;
export type RtlAnchor = z.infer<typeof rtlSchema>;
